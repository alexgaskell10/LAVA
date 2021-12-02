from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
from math import floor
from copy import deepcopy
from random import shuffle
import numpy as np
import multiprocessing

from transformers.tokenization_auto import AutoTokenizer
import wandb

import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from transformers import AutoModel

from .utils import (
    safe_log, right_pad, batch_lookup, EPSILON, make_dot, set_dropout, one_hot, lmap, lfilter, print_results, gs, 
    timing
)
from .ruletaker.theory_label_generator import call_theorem_prover_from_lst

import logging

torch.manual_seed(0)
np.random.seed(0)

@Model.register("adversarial_base")
class AdversarialGenerator(Model):
    def __init__(self,
        qa_model: Model,
        variant: str,
        vocab: Vocabulary = None,
        num_labels: int = 2,
        regularizer: Optional[RegularizerApplicator] = None,
        dataset_reader = None,
        num_monte_carlo = 1,
        add_NAF = False,
        word_overlap_scores = False,
        benchmark_type = "random",
        bernoulli_node_prediction_level = 'node-level',
        adversarial_perturbations = '',
        max_flips = -1,
        **kwargs,
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.variant = variant
        self.qa_model = qa_model
        self.qa_model._loss = nn.CrossEntropyLoss(reduction='none')
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self.qa_vocab = qa_model.vocab
        self.dataset_reader = dataset_reader
        self.vocab = vocab
        self.regularizer = regularizer
        self.num_labels = num_labels
        self.word_overlap_scores = word_overlap_scores

        if bernoulli_node_prediction_level == 'sequence-level':
            self.gen_model = GenerativeBaselineNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, num_labels=2, dropout=0.1)
        elif bernoulli_node_prediction_level == 'node-level':
            self.gen_model = GenerativeNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, has_naf=add_NAF, num_labels=2, dropout=0.1)

        self._p0 = torch.tensor(-float(1e9))        # TODO: set dynamically
        self.n_mc = num_monte_carlo          # Number of monte carlo steps
        self._logprob_method = 'CE'      # TODO: set dynamically
        self._z_mask = -1

        self.answers = {}
        self.records = []
        self.prev_answers = {'train': [], 'val': []}
        
        self.alpha = 0.8        # TODO: some hparam validation on this
        self.c = 0
        self.v = 0
        self.reinforce_baseline = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        set_dropout(self.gen_model, 0.0)
        set_dropout(self.qa_model, 0.0)

        self.benchmark_type = None if benchmark_type == 'none' else benchmark_type
        if self.benchmark_type == 'random':
            self.benchmark = self.random_benchmark
        elif self.benchmark_type == 'word_score':
            self.benchmark = self.wordscore_benchmark
            self.all_word_overlap_scores = torch.tensor(self.dataset_reader._word_overlap_scores_lst)
            self.mean_overlap_score = self.all_word_overlap_scores.mean()
        
        # Turn off verbose logging for problog
        logging.getLogger('problog').setLevel(logging.WARNING)

        self.adv_perturbations = adversarial_perturbations.split(',')

        self.wandb = os.environ['WANDB_LOG'] == 'true'
        self.max_flips = max_flips

    # @timing
    def forward(self, phrase=None, label=None, metadata=None, retrieval=None, word_overlap_scores=None, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. 

            - p(e|z,q) - answering term (qa_model)
            - q(z|e,q,c) - variational distribution (infr_model)
            - p(z|q,c) - generative distribution (gen_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
        '''
        self._n_mc = self.n_mc if self.training else 1
        if not self.training:
            set_dropout(self.gen_model, 0.0)
        self._d = phrase['tokens']['token_ids'].device
        qlens = [m['QLen'] for m in metadata]
        qlens_ = [m['QLen'] for m in metadata for _ in range(self._n_mc)]
        lens = torch.tensor(qlens_, device=self._d).unsqueeze(1)
        metadata_orig = [m for m in metadata for _ in range(self._n_mc)]
        orig_sentences = [[m['question_text']] + m['context'].split('.') for m in metadata]
        meta_records = [m['meta_record'] for m in metadata]
        meta_records_orig = [m for m in meta_records for _ in range(self._n_mc)]

        max_facts = max(len(meta['fact_indices']) for meta in metadata)
        max_rules = max(len(meta['rule_indices']) for meta in metadata)
        fact_idxs = torch.tensor([n['fact_indices'] + [-1]*(max_facts-len(n['fact_indices'])) for n in metadata], device=self._d)
        rule_idxs = torch.tensor([n['rule_indices'] + [-1]*(max_rules-len(n['rule_indices'])) for n in metadata], device=self._d)
        
        polarity = torch.tensor([1-int('not' in m['question_text']) for m in metadata], device=self._d)

        nodes = [torch.tensor(m['node_label'][:-1]).nonzero().squeeze(1) for m in metadata]
        max_nodes = max(len(n) for n in nodes)
        nodes = torch.tensor([n.tolist() + [-1]*(max_nodes-len(n)) for n in nodes], device=self._d)
        proof_sentences = [[s[0]]+[s[i+1] for i in n if i!=-1] for n,s in zip(nodes, orig_sentences)]

        # Obtain retrieval logits
        ques_logits, sent_logits, eqiv_logits = self.gen_model(phrase)

        # Take multiple monte carlo samples by tiling the logits
        sent_logits_ = sent_logits.repeat_interleave(self._n_mc, dim=0)
        ques_logits_ = ques_logits.repeat_interleave(self._n_mc, dim=0)
        eqiv_logits_ = eqiv_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)
        polarity_ = polarity.repeat_interleave(self._n_mc, dim=0)
        nodes_ = nodes.repeat_interleave(self._n_mc, dim=0)
        fact_idxs_ = fact_idxs.repeat_interleave(self._n_mc, dim=0)
        rule_idxs_ = rule_idxs.repeat_interleave(self._n_mc, dim=0)

        z_sent, z_1hot_sent = self._draw_samples(sent_logits)        # (bsz * mc steps, num retrievals)
        z_ques, z_1hot_ques = self._draw_samples(ques_logits)        # (bsz * mc steps, num retrievals)
        z_eqiv, z_1hot_eqiv = self._draw_samples(eqiv_logits)        # (bsz * mc steps, num retrievals)
        
        # Adjust senteqiv samples here
        maskers = [rule_idxs_, z_sent] if 'sentence_elimination' in self.adv_perturbations else [rule_idxs_, z_sent]
        z_eqiv = self.mask_draws(z_eqiv, *maskers)
        if self.max_flips > 0:
            z_eqiv = z_eqiv.topk(self.max_flips, -1).values

        meta_records_ = meta_records_orig
        metadata_ = metadata_orig
        if 'sentence_elimination' in self.adv_perturbations:
            meta_records_ = self.update_meta_sentelim(z_sent, meta_records_, metadata_)
        if 'question_flip' in self.adv_perturbations:
            meta_records_ = self.update_meta_quesflip(z_ques, meta_records_, metadata_)
        if 'equivalence_substitution' in self.adv_perturbations:
            meta_records_ = self.update_meta_eqivsubt(z_eqiv, meta_records_, metadata_)

        engine_labels = self.run_tp(meta_records_)
        if self.training:
            new_labels = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels, label_)]     # This is a hack to catch cases where problog cannot solve the logic program. Only happens v. rarely though so not an issue
            skip_ids = []
        else:
            new_labels = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels, label_)]
            skip_ids = [n for n,e in enumerate(engine_labels) if e is None]
        modified_label_ = 1 - torch.tensor(new_labels, device=self._d, dtype=torch.int)

        batch = self._prep_batch(meta_records_, modified_label_)
        new_metadata = batch['metadata']
        if True:
            # Verify meta records sentences are in metadata
            records_sentences = [[i['text'] for i in list(record['triples'].values()) + list(record['rules'].values()) if not i.get('mask',0)] for record in meta_records_]
            assert all([all([sent in meta['context'] for sent in sentences]) for meta, sentences in zip(new_metadata, records_sentences)])
            # Verify all metadata sentences are in meta records
            assert all([all([(sent+'.').strip() in sentences for sent in meta['context'].split('.')[:-1]]) for meta, sentences in zip(new_metadata, records_sentences)])

        with torch.no_grad():
            qa_output = self.qa_model(**batch)
        qa_logprobs = -qa_output['loss']
        qa_probs = qa_logprobs.exp()
        probs = qa_output["label_probs"]
        preds = probs.argmax(-1)
        qa_logits = qa_output['label_logits']
        # qa_logprobs = torch.where(torch.isnan(qa_logprobs), torch.tensor(0., device=self._d), qa_logprobs)

        sent_logprobs, ques_logprobs, eqiv_logprobs = 0, 0, 0
        if 'sentence_elimination' in self.adv_perturbations:
            sent_logprobs = self._compute_logprobs(z_1hot_sent, sent_logits_)
        if 'question_flip' in self.adv_perturbations:
            ques_logprobs = self._compute_logprobs(z_1hot_ques, ques_logits_)
        if 'equivalence_substitution' in self.adv_perturbations:
            eqiv_logprobs = self._compute_logprobs(z_1hot_eqiv, eqiv_logits_)        
        
        gen_logprobs = sent_logprobs + ques_logprobs + eqiv_logprobs

        # Compute objective function
        l = qa_logprobs.detach()      # Using Jensen's inquaity
        l[torch.isnan(l)] = 0

        # Update the learning signal statistics
        cb = torch.mean(l)
        vb = torch.var(l)
        self.c = self.alpha * self.c + (1-self.alpha) * cb
        self.v = self.alpha * self.v + (1-self.alpha) * vb

        l = (l - self.c) / max(1, self.v)
        reinforce_reward = torch.mul(l.detach(), gen_logprobs)
        baseline_error = l - self.reinforce_baseline
        baseline_term = -torch.pow(baseline_error, 2) #torch.mul(l.detach(), torch.pow(baseline_error, 2))      # Didn't work when rescaling by l as outlined in the original paper...

        estimator = reinforce_reward + baseline_term
        # TODO: explore adding KL term for prior: https://math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution
        aux_signals = 0

        assert not torch.isnan(estimator).any()
        outputs = {"loss": -(estimator.mean() + aux_signals), "estimator": estimator.mean()}
 
        correct_bl = None
        if self.benchmark_type != None: 
            qa_output_baseline, z_bl, z_bl_onehot, batch_bl, meta_records_bl = self.benchmark(
                sent_logits=sent_logits,
                ques_logits=ques_logits,
                eqiv_logits=eqiv_logits,
                metadata=metadata,
                label=label,
                polarity=polarity,
                nodes=nodes,
                word_overlap_scores=word_overlap_scores,
                meta_records=meta_records,
            )

            new_metadata_bl = batch_bl['metadata']
            engine_labels_bl = call_theorem_prover_from_lst(instances=meta_records_bl)
            new_labels_bl = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels_bl, label)]     # TODO: this is a hack to catch cases where problog cannot solve the logic program. Find a better solution here.
            modified_label_bl = 1 - torch.tensor(new_labels_bl, device=self._d, dtype=torch.int)

            if True:
                # Verify meta records sentences are in metadata
                records_sentences_bl = [[i['text'] for i in list(record['triples'].values()) + list(record['rules'].values()) if not i.get('mask',0)] for record in meta_records_bl]
                assert all([all([sent in meta['context'] for sent in sentences]) for meta, sentences in zip(new_metadata_bl, records_sentences_bl)])
                # Verify all metadata sentences are in meta records
                assert all([all([(sent+'.').strip() in sentences for sent in meta['context'].split('.')[:-1]]) for meta, sentences in zip(new_metadata_bl, records_sentences_bl)])

                correct_bl = (qa_output_baseline["label_probs"].argmax(-1) == modified_label_bl)
        
        correct = (preds == modified_label_)
        correct[skip_ids] = False
        log_metrics = {}

        if True:
            sentences = [[m['question_text']] + m['context'].split('.') for m in new_metadata]
            log_metrics['lens'] = np.mean([len(s) for s in sentences])
            records = []
            for i in range(len(correct)):
                i_ = floor(i / self._n_mc)
                record = {
                    "id": metadata_[i_]['id'],
                    "qa_fooled": correct[i].item(),
                    "label": label[i_].item(),
                    "mod_label": modified_label_[i].item(),     # Target label (i.e. 1-correct label)
                    "polarity": polarity[i_].item(),
                    "question": sentences[i][0],
                    "orig_sentences": orig_sentences[i_],
                    "sampled_sentences": sentences[i],
                    "proof_sentences": proof_sentences[i_][1:],
                    "qa_probs": probs[i].cpu().tolist(),
                    "qa_preds": preds[i].item(),
                    "z_sent": z_sent[i].tolist() if z_sent is not None else None,
                    "z_ques": z_ques[i].tolist() if z_ques is not None else None,
                    "z_eqiv": z_eqiv[i].tolist() if z_eqiv is not None else None,
                    "orig_proof_depth": metadata_[i]["QDep"],
                }
                records.append(record)
                record["label"], record["polarity"], record["mod_label"], record["qa_probs"]
            self.records.extend(records)

        self.log_results(qlens_, correct, outputs, ref=correct_bl, log_metrics=log_metrics)

        return outputs

    def mask_draws(self, tensor, *targets):
        ''' Mask values in tensor if they appear in any
            of the targets.
        '''
        tgt = torch.cat(targets, dim=1)
        mask = 1 - torch.prod(torch.cat([(tensor != tgt[:,i:i+1]).unsqueeze(0) for i in range(tgt.size(1))], dim=0), dim=0)
        tensor = torch.where(mask.bool(), torch.tensor(self._z_mask, device=self._d), tensor)
        max_flips = (tensor != self._z_mask).sum(-1).max()
        tensor = tensor.sort(-1, descending=True).values[:,:max_flips]
        return tensor

    # @timing
    def update_meta_sentelim(self, z, meta_records_, metadata_):
        meta_records = [deepcopy(m) for m in meta_records_]
        metadata = [deepcopy(m) for m in metadata_]
        for n, (nodes, meta) in enumerate(zip(z, metadata)):
            qid = 'Q' + meta['id'].split('-')[-1]
            meta_records[n]['questions'] = {qid: meta_records[n]['questions'][qid]}
            sentence_scramble = meta['sentence_scramble']
            n_sents = len(sentence_scramble)
            scrambled_nodes = [sentence_scramble[i] for i in nodes if i not in [self._z_mask, n_sents]]
            nfacts = len(meta_records[n]['triples'])
            nrules = len(meta_records[n]['rules'])
            assert meta_records[n]['questions'][qid]['question'] == meta['question_text']
            for i in range(1, nfacts+1):
                meta_records[n]['triples'][f'triple{i}']['mask'] = 0 if i in scrambled_nodes else 1
            for i in range(1, nrules+1):
                meta_records[n]['rules'][f'rule{i}']['mask'] = 0 if i + nfacts in scrambled_nodes else 1
            continue
        return meta_records

    # @timing
    def update_meta_quesflip(self, z, meta_records_, metadata_):
        meta_records = [deepcopy(m) for m in meta_records_]
        metadata = [deepcopy(m) for m in metadata_]
        for n, (nodes, meta) in enumerate(zip(z, metadata)):
            qid = 'Q' + meta['id'].split('-')[-1]
            meta_records[n]['questions'] = {qid: meta_records[n]['questions'][qid]}
            if nodes == -1:
                meta_records[n]['questions'][qid]['orig_question'] = meta_records[n]['questions'][qid]['question']
                continue
            qtext_new = meta_records[n]['questions'][qid]['flipped_ques']
            qtext_old = meta_records[n]['questions'][qid]['question']
            qrepr_old = meta_records[n]['questions'][qid]['representation']
            qrepr_new = qrepr_old.replace("+","-") if "+" in qrepr_old else qrepr_old.replace("-","+")
            meta_records[n]['questions'][qid]['question'] = qtext_new
            meta_records[n]['questions'][qid]['representation'] = qrepr_new
            meta_records[n]['questions'][qid]['orig_question'] = qtext_old
            metadata[n]['question_text'] = qtext_new
        return meta_records

    # @timing
    def update_meta_eqivsubt(self, z, meta_records_, metadata_):
        meta_records = [deepcopy(m) for m in meta_records_]
        metadata = [deepcopy(m) for m in metadata_]
        for n, (nodes, meta) in enumerate(zip(z, metadata)):
            qid = 'Q' + meta['id'].split('-')[-1]
            meta_records[n]['questions'] = {qid: meta_records[n]['questions'][qid]}
            sentence_scramble = meta['sentence_scramble']
            nfacts = len(meta_records[n]['triples'])
            nrules = len(meta_records[n]['rules'])
            scrambled_nodes = [sentence_scramble[i] for i in nodes if i not in [-1]]
            scrambled_facts = [i for i in scrambled_nodes if i <= nfacts]
            meta_records[n]['equivalence_substitution'] = {}

            fact_tracker, rule_tracker, popped_rules = nfacts, nrules, []
            for fact_id in scrambled_facts:
                if fact_tracker <= 3:
                    continue
                factid = 'triple' + str(fact_id)
                ruleid = 'rule' + str(rule_tracker + 1)
                head_fact = meta_records[n]['triples'].pop(factid)
                head_fact['text'] = head_fact['text'].replace('The','the')
                popped_rules.append(factid)
                body_factids, body_reprs, body_texts = [], [], []
                for i in np.random.permutation(range(nfacts)):
                    body_factid = 'triple'+str(i+1)
                    if body_factid in popped_rules or len(body_factids) == min(2, fact_tracker):
                        continue
                    body_repr = meta_records[n]['triples'][body_factid]['representation']
                    body_text = meta_records[n]['triples'][body_factid]['text']
                    body_text = body_text.replace('The','the')
                    body_factids.append(body_factid)
                    body_reprs.append(body_repr)
                    body_texts.append(body_text.rstrip('.'))

                assert body_reprs and body_texts
                body_repr = '(' + ' '.join(body_reprs) + ')'
                body_text = ' and '.join(body_texts)
                repr = '(' + body_repr + ' -> ' + head_fact['representation'] + ')'
                text = f"If {body_text} then {head_fact['text']}"

                meta_records[n]['rules'][ruleid] = {'text': text, 'representation': repr, 'mask': 0}
                meta_records[n]['equivalence_substitution'][ruleid] = body_factids

                fact_tracker -= 1
                rule_tracker += 1

        return meta_records

    def _compute_logprobs(self, z, logits):
        ''' Compute the log probability of sample z from
            (unnormalized) distribution logit.
        '''
        if self._logprob_method == 'CE':
            mask = 1 - (z==-1).int()
            y = (z + 1 - mask).flatten()
            logits_ = logits.view(-1,2)
            logprobs_ = -self._loss(logits_, y)
            logprobs = logprobs_.view(z.shape)
            return (logprobs * mask).sum(-1) / mask.sum(-1)
        else:
            raise NotImplementedError

    def log_results(self, qlens, acc, outputs, log_metrics, tp=None, ref=None):
        if tp is None:
            tp = torch.full_like(acc, False)
        if ref is None:
            ref = torch.full_like(acc, False)

        for n, (d, c, r, t) in enumerate(zip(qlens, acc, ref.repeat_interleave(self._n_mc, dim=0), tp)):
            if d not in self.answers:
                self.answers[d] = [[], [], []]
            self.answers[d][0].append(c.item())
            self.answers[d][1].append(r.item())
            if t != -1:
                self.answers[d][2].append(t)

        if self.training and not self.wandb:
            print_results(self.answers, self._n_mc)

        if self.wandb:
            self.log_wandb(qlens, acc, outputs, log_metrics)

    def log_wandb(self, qlens, acc, outputs, metrics):
        prefix = 'tr_' if self.training else 'val_'
        grouped = {qlen: [a.item() for a,q in zip(acc, qlens) if q==qlen] for qlen in qlens}
        grouped['all'] = acc.tolist()
        metrics = {
            **{prefix + str(k): v for k,v in metrics.items()},
            **{prefix + 'acc_' + str(k): v.count(True) / len(v) for k,v in grouped.items()},
            **{prefix + 'N_' + str(k): len(v) for k,v in grouped.items()},
            **{prefix + k: v for k,v in outputs.items()},
        }
        wandb.log(metrics)

    def _prep_batch(self, meta_records, label):
        sentences = []
        for meta_record, e in zip(meta_records, label):
            qid = list(meta_record['questions'].keys())[0]
            question = meta_record['questions'][qid]['question']
            context = [i['text'] for i in list(meta_record['triples'].values()) + list(meta_record['rules'].values()) if not i.get('mask')]
            shuffle(context)
            sentences.append((question, ' '.join(context).strip(), e))

        batch = self.dataset_reader.encode_batch(sentences, self.qa_vocab, disable=True)
        return self.dataset_reader.move(batch, self._d)

    def _draw_samples(self, logits, random_chance=False):
        ''' Obtain samples from a distribution
            - p: probability distribution

            random_chance: sample from uniform distribution
        '''
        if random_chance:
            # Overwrite all weights with uniform weight (for masked positions)
            logits = torch.where(logits != self._p0, torch.zeros_like(logits), self._p0.to(self._d))
        else:
            # Duplicate tensor for multiple monte carlo samples
            logits = logits.repeat_interleave(self._n_mc, dim=0)        # (bsz * mc steps, num retrievals)
        
        logits_ = logits.view(-1, logits.size(-1))

        # Obtain samples
        # During validation, use argmax unless prob = 0.5, in which case sample
        draws = gs(logits_).argmax(-1)
        # if self.training:
        if True:
            samples = draws
        else:
            greedy = logits_.argmax(-1)
            samples = torch.where((logits_ == 0).all(-1), draws, greedy)

        # Ensure padding sentences are not sampled
        mask = (logits_ == self._p0).all(-1).int()
        samples *= 1 - mask
        samples = samples.view(logits.shape[:-1])

        max_draws = max(samples.sum(-1).max(), 1)
        z = torch.full_like(samples, -1)[:,:max_draws]
        row, idx = samples.nonzero(as_tuple=True)
        counts = row.bincount()
        for n in range(z.size(0)):
            if n not in row:
                continue
            count = counts[n]
            z[n, :count] = idx[:count]
            idx = idx[count:]

        # Replace padding sentences with -1
        samples -= mask.view(samples.shape)

        assert z.numel() > 0

        return z, samples

    def _draw_samples_wordscore(self, logits, word_overlap_scores, random_chance=False):
        ''' Obtain samples from a distribution
            - p: probability distribution

            random_chance: sample from uniform distribution
        '''
        mask = (logits == self._p0).all(-1).int()

        tmp = torch.zeros_like(mask, dtype=torch.float)
        tmp[:,:word_overlap_scores.size(1)] = word_overlap_scores
        logits_ = tmp
        scores = torch.clamp(0.5 + logits_ - self.mean_overlap_score, 0, 1)
        samples = torch.bernoulli(scores)
        samples *= 1 - mask       # Ensure padding sentences are not sampled

        # logits_ = logits.view(-1, logits.size(-1))
        # samples = gs(logits_).argmax(-1)
        # mask = (logits_ == -1e9).all(-1).int()
        # samples *= 1 - mask       # Ensure padding sentences are not sampled
        # samples = samples.view(logits.shape[:-1])

        max_draws = samples.sum(-1).max().int()
        z = torch.full_like(samples, -1)[:,:max_draws]
        row, idx = samples.nonzero(as_tuple=True)
        counts = row.bincount()
        for n in range(z.size(0)):
            if n not in row:
                continue
            count = counts[n]
            z[n, :count] = idx[:count]
            idx = idx[count:]

        # Replace padding sentences with -1
        samples = samples * (1-mask.view(samples.shape)) - mask.view(samples.shape)

        return z, samples

    def decode(self, i, batch):
        return self.dataset_reader.decode(batch['phrase']['tokens']['token_ids'][i]).split('.')

    def random_benchmark(self, sent_logits=False, ques_logits=False, eqiv_logits=False, metadata=False, label=False, polarity=False, nodes=False, meta_records=False, **kwargs):
        with torch.no_grad():
            z_sent_bl, z_sent_bl_onehot = self._draw_samples(sent_logits, random_chance=True)
            z_ques_bl, z_ques_bl_onehot = self._draw_samples(ques_logits, random_chance=True)
            z_eqiv_bl, z_eqiv_bl_onehot = self._draw_samples(eqiv_logits, random_chance=True)
            if 'sentence_elimination' not in self.adv_perturbations:
                z_sent_bl = None

            meta_records_bl = meta_records
            metadata_bl = metadata
            if 'sentence_elimination' in self.adv_perturbations:
                meta_records_bl = self.update_meta_sentelim(z_sent_bl, meta_records_bl, metadata_bl)
            if 'question_flip' in self.adv_perturbations:
                meta_records_bl = self.update_meta_quesflip(z_ques_bl, meta_records_bl, metadata_bl)
            if 'equivalence_substitution' in self.adv_perturbations:
                meta_records_bl = self.update_meta_eqivsubt(z_eqiv_bl, meta_records_bl, metadata_bl)

            batch_bl = self._prep_batch(meta_records_bl, label)
            qa_output_baseline = self.qa_model(**batch_bl)

        return qa_output_baseline, z_sent_bl, z_sent_bl_onehot, batch_bl, meta_records_bl

    def wordscore_benchmark(self, sent_logits=False, metadata=False, label=False, polarity=False, nodes=False, word_overlap_scores=False, **kwargs):
        with torch.no_grad():
            z_baseline, z_bl_onehot = self._draw_samples_wordscore(sent_logits, word_overlap_scores, random_chance=True)
            batch_baseline = self._prep_batch(z_baseline, metadata, label)
            qa_output_baseline = self.qa_model(**batch_baseline)

        return qa_output_baseline, z_baseline, z_bl_onehot, batch_baseline

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if len(self.answers) == 0:
            self._precomputed_metrics.pop('predictions', None)
            return self._precomputed_metrics

        prefix = 'tr_' if self.training else 'val_'
        grouped = {k: v[0] for k,v in self.answers.items()}
        grouped['all'] = [v for lst in grouped.values() for v in lst]

        metrics = {
            'EM': grouped['all'].count(True) / len(grouped['all']),
            **{prefix + 'acc_' + str(k): v.count(True) / len(v) for k,v in grouped.items()},
            **{prefix + 'N_' + str(k): len(v) for k,v in grouped.items()},
        }

        self._precomputed_metrics = metrics

        if reset:
            print_results(self.answers, self._n_mc)
            metrics['predictions'] = self.records
            self.answers = {}
            self.records = []

        return metrics

    # @timing
    def run_tp(self, meta_records):
        return call_theorem_prover_from_lst(instances=meta_records)


class _BaseSentenceClassifier(Model):
    def __init__(self, variant, vocab, dataset_reader, has_naf, regularizer=None, num_labels=1, dropout=0.):
        super().__init__(vocab, regularizer)
        self._predictions = []
        self._dropout = dropout

        self.variant = variant
        self.dataset_reader = dataset_reader
        self.model = AutoModel.from_pretrained(variant)
        assert 'roberta' in variant     # Only implemented for roberta currently

        transformer_config = self.model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self.model.config.hidden_size

        # unifing all model classification layer
        self.sent_classifier = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.ques_classifier = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.eqiv_classifier = NodeClassificationHead(self._output_dim, dropout, num_labels)
        self.naf_layer = nn.Linear(self._output_dim, self._output_dim)
        self.has_naf = has_naf

        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()
        self.num_labels = num_labels

        # Split sentences in the context based on full stop
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')

        self._p0 = torch.tensor(-float(1e9))

    def forward(self, x) -> torch.Tensor:
        ''' Forward pass of the network. Outputs a distribution over sentences z. 

            - q(z|e,q,c) - variational distribution (infr_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
            - q: claim
        '''
        # Compute representations
        embs = self.model(x)[0]
        cls_emb = embs[:,0,:]
        naf_repr = self.naf_layer(cls_emb)

        max_num_sentences = (x == self.split_idx).nonzero()[:,0].bincount().max()
        batch_sent_logits = torch.full((x.size(0), max_num_sentences, self.num_labels), self._p0).to(self._d)       # shape: (bsz, # sentences, 2)
        batch_eqiv_logits = batch_sent_logits.clone()[:,1:,:]                                                               # shape: (bsz, # sentences - 1, 2)   2nd dim is smaller than above because it doesn't need NAF reprs
        batch_ques_logits = batch_sent_logits.clone()[:,:1,:]
        for b in range(x.size(0)):
            # Create list of end idxs of each context item
            end_idxs = (x[b] == self.split_idx).nonzero().squeeze(1).tolist()
            q_end = end_idxs[0]
            q_start = 2             # Beginning of question ix. Ignores BOS tokens
            start_idxs = [q_start, q_end + 4] + end_idxs[1:-1]       # +4 because tok adds four "decorative" tokens at the beginning of the context
            offsets = list(zip(start_idxs, end_idxs))

            # Form tensor containing sentence-level mean of token embeddings
            n_sentences = len(offsets)
            reprs = torch.zeros(n_sentences, 1, embs.size(-1)).to(self._d)      # shape: (# context items, 1, model_dim)
            for i in range(n_sentences):
                start_idx = offsets[i][0] + 1               # +1 to skip full stop at beginning
                end_idx = offsets[i][1] + 1                 # +1 to include full stop at end
                
                # Extract reprs for tokens in the sentence from the original encoded sequence
                reprs[i, 0] = embs[b, start_idx:end_idx].mean(dim=0)

            # Pass through classifier and store results
            ques_reprs = reprs.view(reprs.size(0), -1)[:1]
            sent_reprs = reprs.view(reprs.size(0), -1)[1:]
            sent_reprs = torch.cat((sent_reprs, naf_repr[b].unsqueeze(0)), 0)
            eqiv_reprs = reprs.view(reprs.size(0), -1)[1:]

            ques_logits = self.ques_classifier(ques_reprs).squeeze(-1)
            sent_logits = self.sent_classifier(sent_reprs).squeeze(-1)
            eqiv_logits = self.eqiv_classifier(eqiv_reprs).squeeze(-1)
            
            batch_ques_logits[b, :len(ques_logits)] = ques_logits
            batch_sent_logits[b, :len(sent_logits)] = sent_logits
            batch_eqiv_logits[b, :len(eqiv_logits)] = eqiv_logits

        return batch_ques_logits, batch_sent_logits, batch_eqiv_logits


class GenerativeNetwork(_BaseSentenceClassifier):
    def forward(self, phrase, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. Outputs a distribution over sentences z. 

            - q(z|q,c) - generative distribution (gen_model)
            - z: retrievals (subset of c)
            - c: context (rules + facts)
            - q: claim
        '''
        qc = phrase['tokens']['token_ids']       # shape = (bsz, context_len)
        self._d = qc.device

        return super().forward(qc)


class GenerativeBaselineNetwork(Model):
    ''' Predicts a single Bernoulli parameter for the whole sample
        (rather than per sentence, as is the case in above GenerativeNetwork)
    '''
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
        super().__init__(vocab, regularizer)

        self.dataset_reader = dataset_reader
        self.model = AutoModel.from_pretrained(variant)
        assert 'roberta' in variant     # Only implemented for roberta currently

        transformer_config = self.model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self.model.config.hidden_size

        self.out_layer = NodeClassificationHead(self._output_dim, 0, num_labels)
        self.num_labels = num_labels
        
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')
        self._p0 = torch.tensor(-float(1e9))

    def forward(self, phrase, **kwargs) -> torch.Tensor:
        
        x = phrase['tokens']['token_ids']
        final_layer_hidden_states = self.model(x)[0]
        out = self.out_layer(final_layer_hidden_states[:,0,:])

        num_sentences = (x == self.split_idx).nonzero()[:,0].bincount()
        node_reprs = torch.cat([out.unsqueeze(1)]*num_sentences.max(), dim=1)
        num_padding = num_sentences.max() - num_sentences

        for n,i in enumerate(num_padding):
            if i == 0:
                continue
            for m in range(1,i+1):
                node_reprs[n,-m] = self._p0

        return node_reprs, None


class NodeClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks.
        TODO: from PRover
    """
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super(NodeClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

