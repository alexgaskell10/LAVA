from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
from math import floor
from copy import deepcopy
from random import shuffle
import random
import numpy as np
import multiprocessing

from transformers import AutoTokenizer
import wandb

import torch
from torch import nn

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.model import remove_pretrained_embedding_params, _DEFAULT_WEIGHTS
from allennlp.nn import util
from allennlp.models.archival import load_archive

from transformers import AutoModel

from ..utils import (
    safe_log, right_pad, batch_lookup, EPSILON, set_dropout, one_hot, lmap, lfilter, print_results, gs, 
    timing
)

from .solver.theory_label_generator import call_theorem_prover_from_lst
from .modules.adversarial_modules import GenerativeBaselineNetwork, GenerativeNetwork


import logging

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

@Model.register("adversarial_base")
class AdversarialGenerator(Model):
    def __init__(self,
        qa_model: Model,
        variant: str,
        vocab: Vocabulary = None,
        num_labels: int = 2,
        regularizer = None,
        dataset_reader = None,
        num_monte_carlo = 1,
        val_num_monte_carlo = 1,
        add_NAF = False,
        benchmark_type = 'random',
        word_overlap_scores = False,
        bernoulli_node_prediction_level = 'node-level',
        adversarial_perturbations = '',
        max_flips = -1,
        max_elims = -1,
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
        self._p0 = torch.tensor(-float(1e9))

        if bernoulli_node_prediction_level == 'sequence-level':
            self.gen_model = GenerativeBaselineNetwork(variant=variant, vocab=vocab, null_probability=self._p0, dataset_reader=dataset_reader, num_labels=2, dropout=0.1)
        elif bernoulli_node_prediction_level == 'node-level':
            self.gen_model = GenerativeNetwork(variant=variant, vocab=vocab, null_probability=self._p0, dataset_reader=dataset_reader, has_naf=add_NAF, num_labels=2, dropout=0.1)

        self.n_mc = num_monte_carlo          # Number of monte carlo steps
        self.val_mc = val_num_monte_carlo          # Number of validation trials per sample
        self._logprob_method = 'CE'
        self._z_mask = -1

        self.answers = {}
        self.records = []
        self.prev_answers = {'train': [], 'val': []}
        
        self.alpha = 0.8
        self.c = 0
        self.v = 0
        self.reinforce_baseline = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        set_dropout(self.gen_model, 0.0)
        set_dropout(self.qa_model, 0.0)
        
        # Turn off verbose logging for problog
        logging.getLogger('problog').setLevel(logging.WARNING)

        self.adv_perturbations = adversarial_perturbations.split(',')

        self.wandb = 'WANDB_LOG_1' in os.environ and os.environ['WANDB_LOG_1'] == 'true'
        self.max_flips = int(max_flips)
        self.max_elims = int(max_elims)

    def forward(self, phrase=None, label=None, metadata=None, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. 

            - p(e|z,q) - answering term (qa_model)
            - q(z|e,q,c) - variational distribution (infr_model)
            - p(z|q,c) - generative distribution (gen_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
        '''
        self._n_mc = self.n_mc if self.training else self.val_mc
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
        z_sent, z_1hot_sent = self._draw_samples(sent_logits)        # (bsz * mc steps, num retrievals)
        z_ques, z_1hot_ques = self._draw_samples(ques_logits)        # (bsz * mc steps, num retrievals)
        z_eqiv, z_1hot_eqiv = self._draw_samples(eqiv_logits)        # (bsz * mc steps, num retrievals)

        # Take multiple monte carlo samples by tiling the logits
        sent_logits_ = sent_logits.repeat_interleave(self._n_mc, dim=0)
        ques_logits_ = ques_logits.repeat_interleave(self._n_mc, dim=0)
        eqiv_logits_ = eqiv_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)
        fact_idxs_ = fact_idxs.repeat_interleave(self._n_mc, dim=0)
        rule_idxs_ = rule_idxs.repeat_interleave(self._n_mc, dim=0)

        # Adjust senteqiv samples here
        if self.max_elims > 0:
            z_sent = z_sent.topk(min(self.max_elims, z_sent.size(1)), 1).values
        maskers = [rule_idxs_, z_sent] if 'sentence_elimination' in self.adv_perturbations else [rule_idxs_]
        z_eqiv = self.mask_draws(z_eqiv, *maskers)
        if self.max_flips > 0:
            z_eqiv = z_eqiv.topk(min(self.max_flips, z_eqiv.size(1)), 1).values

        meta_records_ = meta_records_orig
        metadata_ = metadata_orig
        if 'sentence_elimination' in self.adv_perturbations:
            meta_records_ = self.update_meta_sentelim(z_sent, meta_records_, metadata_)
        else:
            meta_records_ = self.update_meta(meta_records_)
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
            records_sentences = [[i['text'] for i in list(record['triples'].values()) + list(record['rules'].values()) if not i.get('mask',1)] for record in meta_records_]
            assert all([all([sent in meta['context'] for sent in sentences]) for meta, sentences in zip(new_metadata, records_sentences)])
            # Verify all metadata sentences are in meta records
            assert all([all([(sent+'.').strip() in sentences for sent in meta['context'].split('.')[:-1]]) for meta, sentences in zip(new_metadata, records_sentences)])

        with torch.no_grad():
            qa_output = self.qa_model(**batch)
        qa_logprobs = -qa_output['loss']
        probs = qa_output["label_probs"]
        preds = probs.argmax(-1)

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
        aux_signals = 0

        # assert not torch.isnan(estimator).any()
        outputs = {"loss": -(estimator.mean() + aux_signals), "estimator": estimator.mean()}
 
        if not self.training:
            # Compute the best sample for each set of MC samples. Retain the best and 
            # discard the rest
            correct_ps = (modified_label_ - probs[:,1]).abs()
            correct_ps[skip_ids] = 1        # Skip samples where the solver couldn't solve for entailment
            correct_ps_ = correct_ps.view(-1,self._n_mc)
            ixs = correct_ps_.argmin(-1) + torch.arange(correct_ps_.size(0), device=self._d) * self._n_mc

            # Apply filtering
            preds = preds[ixs]
            modified_label_ = modified_label_[ixs]
            z_sent = z_sent[ixs]
            z_eqiv = z_eqiv[ixs]
            z_ques = z_ques[ixs]
            new_metadata = [m for n,m in enumerate(new_metadata) if n in ixs]
            metadata_ = [m for n,m in enumerate(metadata_) if n in ixs]
            probs = probs[ixs]
            qlens_ = [q for n,q in enumerate(qlens_) if n in ixs]

            correct = (preds == modified_label_)
        else:
            correct = (preds == modified_label_)
            correct[skip_ids] = False
            
        log_metrics = {}
        log_metrics['n_queselim'] = (z_sent != -1).sum(-1).float().mean().item()
        log_metrics['n_eqivsubt'] = (z_eqiv != -1).sum(-1).float().mean().item()
        log_metrics['n_quesflip'] = (z_ques != -1).sum(-1).float().mean().item()

        if True:
            sentences = [[m['question_text']] + m['context'].split('.') for m in new_metadata]
            log_metrics['lens'] = np.mean([len(s) for s in sentences])
            records = []
            for i in range(len(correct)):
                i_ = floor(i / self._n_mc) if self.training else i
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
            self.records.extend(records)

        self.log_results(qlens_, correct, outputs, log_metrics=log_metrics)

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

    def update_meta(self, meta_records_):
        meta_records = [deepcopy(m) for m in meta_records_]
        for n in range(len(meta_records)):
            record = meta_records[n]
            nfacts = len(record['triples'])
            nrules = len(record['rules'])
            for i in range(1, nfacts+1):
                meta_records[n]['triples'][f'triple{i}']['mask'] = 0
            for i in range(1, nrules+1):
                meta_records[n]['rules'][f'rule{i}']['mask'] = 0
            continue
        return meta_records

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
                meta_records[n]['triples'][f'triple{i}']['mask'] = 1 if i in scrambled_nodes else 0
            for i in range(1, nrules+1):
                meta_records[n]['rules'][f'rule{i}']['mask'] = 1 if i + nfacts in scrambled_nodes else 0
            continue
        return meta_records

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
                repr = '(' + body_repr + ' -> k' + head_fact['representation'] + ')'
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
        ref = torch.full_like(acc, False)

        for n, (d, c, r, t) in enumerate(zip(qlens, acc, ref, tp)):
            if d not in self.answers:
                self.answers[d] = [[], [], []]
            self.answers[d][0].append(c.item())
            self.answers[d][1].append(r.item())
            if t != -1:
                self.answers[d][2].append(t)

        # if self.training and not self.wandb:
        #     print_results(self.answers, self._n_mc)

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
        # draws = torch.multinomial(logits_.softmax(-1), 1).squeeze(1)
        draws = gs(logits_).argmax(-1)          # Slightly more stable than above
        samples = draws

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

    def decode(self, i, batch):
        return self.dataset_reader.decode(batch['phrase']['tokens']['token_ids'][i]).split('.')

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

    def run_tp(self, meta_records):
        return call_theorem_prover_from_lst(instances=meta_records)

    @classmethod
    def _load(cls,
        config: Params,
        serialization_dir: str,
        weights_file: Optional[str] = None,
        cuda_device: int = -1,
        opt_level: Optional[str] = None,
    ) -> "Model":
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        model_params = config.get("model")

        training_params = config.get("trainer", Params({}))
        opt_level = opt_level or training_params.get("opt_level")

        archive = load_archive(
            archive_file=config.pop("ruletaker_archive"), 
            cuda_device=cuda_device if cuda_device !=-1 else training_params.get("cuda_device"))
        model_params.params['qa_model'] = archive.model.eval()

        reader_type = config.params["dataset_reader"].pop("type")
        if reader_type == 'retriever_reasoning':
            from ..dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
        else:
            raise ValueError
        subs = {'False':False, 'True':True, 'None':None}
        config.params['dataset_reader'] = {k:subs[v] if isinstance(v, str) and v in subs else v for k,v in config.params['dataset_reader'].items()}
        dset = DataReader(**config.get("dataset_reader"))
        model_params.params['dataset_reader'] = dset

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        model = cls(vocab=vocab, **model_params.params)
        # model = Model.from_params(vocab=vocab, params=model_params)


        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        # If vocab+embedding extension was done, the model initialized from from_params
        # and one defined by state dict in weights_file might not have same embedding shapes.
        # Eg. when model embedder module was transferred along with vocab extension, the
        # initialized embedding weight shape would be smaller than one in the state_dict.
        # So calling model embedding extension is required before load_state_dict.
        # If vocab and model embeddings are in sync, following would be just a no-op.
        model.extend_embedder_vocab()

        model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
        model.load_state_dict(model_state)

        return model

