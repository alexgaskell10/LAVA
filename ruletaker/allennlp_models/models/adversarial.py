from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
from math import floor
from copy import deepcopy

from transformers.tokenization_auto import AutoTokenizer
import wandb
import random

import torch
from torch.nn.modules.linear import Linear
from torch import batch_norm_gather_stats_with_counts, nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from transformers import AutoModel

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)
from .utils import safe_log, right_pad, batch_lookup, EPSILON, make_dot, set_dropout, one_hot, lmap, lfilter
from .transformer_binary_qa_model import TransformerBinaryQA
from .baseline import Baseline
from .ruletaker.theory_label_generator import call_theorem_prover_from_lst

import logging
# logging.basicConfig(level=logging.CRITICAL)

torch.manual_seed(0)

@Model.register("adversarial_base")
class AdversarialGenerator(Model):
    def __init__(self,
        qa_model: Model,
        variant: str,
        vocab: Vocabulary = None,
        # pretrained_model: str = None,
        requires_grad: bool = True,
        transformer_weights_model: str = None,
        num_labels: int = 2,
        predictions_file=None,
        layer_freeze_regexes: List[str] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        topk: int = 5,
        sentence_embedding_method: str = 'mean',
        dataset_reader = None,
        num_monte_carlo = 1,
        do_mask_z = True,
        baseline_type = 'decay',
        additional_qa_training = False,
        objective = 'NVIL',
        sampling_method = 'multinomial',
        infr_supervision = False,
        add_NAF = False,
        threshold_sampling = False,
        word_overlap_scores = False,
        benchmark_type = "random",
        bernoulli_node_prediction_level = 'node-level',
        adversarial_perturbations = '',
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.variant = variant
        self.qa_model = qa_model
        self.qa_model._loss = nn.CrossEntropyLoss(reduction='none')
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self._mlloss = nn.BCEWithLogitsLoss(reduction='none')
        self.qa_vocab = qa_model.vocab
        self.dataset_reader = dataset_reader
        if bernoulli_node_prediction_level == 'sequence-level':
            self.gen_model = GenerativeBaselineNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, num_labels=2)
        elif bernoulli_node_prediction_level == 'node-level':
            self.gen_model = GenerativeNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, has_naf=add_NAF, num_labels=2)
        self.vocab = vocab
        self.regularizer = regularizer
        self.sentence_embedding_method = sentence_embedding_method
        self.num_labels = num_labels
        # self.objective = objective
        self.word_overlap_scores = word_overlap_scores
        # self.objective_fn = getattr(self, objective.lower())
        
        self._n_z = topk        # Number of latent retrievals
        self._beta = 0.1        # Scale factor for KL_div TODO: set dynamically
        self._p0 = torch.tensor(-float(1e9))        # TODO: set dynamically
        self.n_mc = num_monte_carlo          # Number of monte carlo steps
        self._logprob_method = 'CE'      # TODO: set dynamically
        self._z_mask = -1
        self._do_mask_z = do_mask_z     # TODO: set dynamically
        # self._additional_qa_training = additional_qa_training
        # self.threshold_sampling = threshold_sampling

        # self._baseline_type = baseline_type
        # if baseline_type == 'Prob-NMN':
        #     self._reinforce = Reinforce(baseline_decay=0.99)
        # elif baseline_type == 'NVIL':
        #     self.baseline_model = BaselineNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, num_labels=1)
        #     self._reinforce = TrainableReinforce(self.baseline_model)
        #     set_dropout(self.baseline_model, 0.0)

        self.answers = {}
        self.prev_answers = {'train': [], 'val': []}
        self._ = 0
        self._smoothing_alpha = 0.1       # TODO: check this
        # self._smoothloss = LabelSmoothingLoss(self._smoothing_alpha)
        self._supervised = infr_supervision
        
        self.alpha = 0.8
        self.c = 0
        self.v = 0
        self.reinforce_baseline = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self._sampler = sampling_method
        
        set_dropout(self.gen_model, 0.0)
        set_dropout(self.qa_model, 0.0)

        self.records = []

        self.benchmark_type = benchmark_type
        if self.benchmark_type == 'random':
            self.benchmark = self.random_benchmark
        elif self.benchmark_type == 'word_score':
            self.benchmark = self.wordscore_benchmark
            self.all_word_overlap_scores = torch.tensor(self.dataset_reader._word_overlap_scores_lst)
            self.mean_overlap_score = self.all_word_overlap_scores.mean()
        
        # Turn off verbose logging for problog
        logging.getLogger('problog').setLevel(logging.WARNING)

        self.adv_perturbations = adversarial_perturbations.split(',')

    def forward_prover(self, phrase=None, label=None, metadata=None, retrieval=None, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. 

            - p(e|z,q) - answering term (qa_model)
            - q(z|e,q,c) - variational distribution (infr_model)
            - p(z|q,c) - generative distribution (gen_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
        '''
        self._d = phrase['tokens']['token_ids'].device
        qlens = [m['QLen'] for m in metadata]
        qlens_ = [m['QLen'] for m in metadata for _ in range(self._n_mc)]
        lens = torch.tensor(qlens_, device=self._d).unsqueeze(1)
        metadata_ = [m for m in metadata for _ in range(self._n_mc)]
        orig_sentences = [[m['question_text']] + m['context'].split('.') for m in metadata]
        
        polarity = torch.tensor([1-int('not' in m['question_text']) for m in metadata], device=self._d)

        nodes = [torch.tensor(m['node_label'][:-1]).nonzero().squeeze(1) for m in metadata]
        max_nodes = max(len(n) for n in nodes)
        nodes = torch.tensor([n.tolist() + [-1]*(max_nodes-len(n)) for n in nodes], device=self._d)
        proof_sentences = [[s[0]]+[s[i+1] for i in n if i!=-1] for n,s in zip(nodes, orig_sentences)]

        # Obtain retrieval logits
        sent_logits = self.gen_model(phrase)

        # Take multiple monte carlo samples by tiling the logits
        sent_logits_ = sent_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)
        polarity_ = polarity.repeat_interleave(self._n_mc, dim=0)
        nodes_ = nodes.repeat_interleave(self._n_mc, dim=0)

        logits, z, z_1hot = self._draw_samples(sent_logits)        # (bsz * mc steps, num retrievals)


        # # Modify labels to reflect adversarial goals
        # # Stick with label if polarity == label and proof nodes not in sampled nodes
        # # Else flip label
        # sampled_proof_nodes = torch.tensor([all([n in _z for n in node if n!=-1]) for _z,node in zip(z, nodes_)], device=self._d)
        # flip_mask = 1 - (polarity_ == label_).int() * (1-sampled_proof_nodes.int())
        # modified_label_ = (label_ != flip_mask).long()

        # batch = self._prep_batch(z, metadata_, modified_label_)
        # with torch.no_grad():
        #     qa_output = self.qa_model(**batch)

        # qa_logprobs = -qa_output['loss']
        # qa_probs = qa_logprobs.exp()
        # probs = qa_output["label_probs"]
        # preds = probs.argmax(-1)
        # qa_logits = qa_output['label_logits']
        # qa_logprobs = -self.qa_model._loss(qa_logits, modified_label_)
        # qa_logprobs = torch.where(torch.isnan(qa_logprobs), torch.tensor(0., device=self._d), qa_logprobs)
        sent_logprobs = self._compute_logprobs(z_1hot, sent_logits_)

        # # Compute objective function
        # # l = qa_logprobs.detach()      # Using Jensen's inquaity
        # l = qa_probs.detach()           # Without Jensen's inquaity
        # l[torch.isnan(l)] = 0

        # # Update the learning signal statistics
        # cb = torch.mean(l)
        # vb = torch.var(l)
        # self.c = self.alpha * self.c + (1-self.alpha) * cb
        # self.v = self.alpha * self.v + (1-self.alpha) * vb

        # l = (l - self.c) / max(1, self.v)
        # reinforce_reward = torch.mul(l.detach(), sent_logprobs)
        # baseline_error = l - self.reinforce_baseline
        # baseline_term = -torch.pow(baseline_error, 2) #torch.mul(l.detach(), torch.pow(baseline_error, 2))      # Didn't work when rescaling by l as outlined in the original paper...

        # estimator = reinforce_reward + baseline_term
        aux_signals = 0
        # aux_signals -= 0.012 * sent_logits.softmax(-1)[:,:,1].mean()
        nodes_onehot = F.one_hot(nodes+1, sent_logits.size(1)+1)[:,:,1:].sum(1)
        # aux_signals += 0.1*self._compute_logprobs(nodes_onehot, sent_logits).mean()
        aux_signals += self._compute_logprobs(nodes_onehot, sent_logits).mean()

        # assert not torch.isnan(estimator).any()
        # outputs = {"loss": -(estimator.mean() + aux_signals)}
        outputs = {"loss": -(aux_signals)}
 
        if True:
            # sentences = [[m['question_text']] + m['context'].split('.') for m in batch['metadata']]
            # correct = (preds == modified_label_)
            correct = torch.tensor([set(i.item() for i in n if i!=-1) == set(i.item() for i in z_ if i!=-1) for n,z_ in zip(nodes_,z)])
        self.log_results(qlens_, correct)

        return outputs

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
        self._d = phrase['tokens']['token_ids'].device
        qlens = [m['QLen'] for m in metadata]
        qlens_ = [m['QLen'] for m in metadata for _ in range(self._n_mc)]
        lens = torch.tensor(qlens_, device=self._d).unsqueeze(1)
        metadata_ = [m for m in metadata for _ in range(self._n_mc)]
        orig_sentences = [[m['question_text']] + m['context'].split('.') for m in metadata]
        meta_records = [m['meta_record'] for m in metadata]
        meta_records_orig = [m for m in meta_records for _ in range(self._n_mc)]
        
        polarity = torch.tensor([1-int('not' in m['question_text']) for m in metadata], device=self._d)

        nodes = [torch.tensor(m['node_label'][:-1]).nonzero().squeeze(1) for m in metadata]
        max_nodes = max(len(n) for n in nodes)
        nodes = torch.tensor([n.tolist() + [-1]*(max_nodes-len(n)) for n in nodes], device=self._d)
        proof_sentences = [[s[0]]+[s[i+1] for i in n if i!=-1] for n,s in zip(nodes, orig_sentences)]

        # Obtain retrieval logits
        ques_logits, sent_logits = self.gen_model(phrase)

        # Take multiple monte carlo samples by tiling the logits
        sent_logits_ = sent_logits.repeat_interleave(self._n_mc, dim=0)
        ques_logits_ = ques_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)
        polarity_ = polarity.repeat_interleave(self._n_mc, dim=0)
        nodes_ = nodes.repeat_interleave(self._n_mc, dim=0)

        _, z_sent, z_1hot_sent = self._draw_samples(sent_logits)        # (bsz * mc steps, num retrievals)
        _, z_ques, z_1hot_ques = self._draw_samples(ques_logits)        # (bsz * mc steps, num retrievals)

        meta_records_ = meta_records_orig
        if 'sentence_elimination' in self.adv_perturbations:
            meta_records_, metadata_ = self.update_meta_sentelim(z_sent, meta_records_, metadata_)
        if 'question_flip' in self.adv_perturbations:
            meta_records_, metadata_ = self.update_meta_quesflip(z_ques, meta_records_, metadata_)

        engine_labels = call_theorem_prover_from_lst(instances=meta_records_)
        new_labels = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels, label_)]     # TODO: this is a hack to catch cases where problog cannot solve the logic program. Find a better solution here.
        modified_label_ = 1 - torch.tensor(new_labels, device=self._d, dtype=torch.int)

        batch = self._prep_batch(z_sent, metadata_, modified_label_)
        if True:
            # Verify batched sentences are same as those in new meta records
            records_sentences = [[i['text'] for i in list(record['triples'].values()) + list(record['rules'].values()) if not i['mask']] for record in meta_records_]
            assert all([all([sent in meta['context'] for sent in sentences]) for meta, sentences in zip(metadata_, records_sentences)])

        with torch.no_grad():
            qa_output = self.qa_model(**batch)

        qa_logprobs = -qa_output['loss']
        qa_probs = qa_logprobs.exp()
        probs = qa_output["label_probs"]
        preds = probs.argmax(-1)
        qa_logits = qa_output['label_logits']
        # qa_logprobs = -self.qa_model._loss(qa_logits, modified_label_)
        qa_logprobs = torch.where(torch.isnan(qa_logprobs), torch.tensor(0., device=self._d), qa_logprobs)
        sent_logprobs = self._compute_logprobs(z_1hot_sent, sent_logits_)

        # Compute objective function
        l = qa_logprobs.detach()      # Using Jensen's inquaity
        l[torch.isnan(l)] = 0

        # Update the learning signal statistics
        cb = torch.mean(l)
        vb = torch.var(l)
        self.c = self.alpha * self.c + (1-self.alpha) * cb
        self.v = self.alpha * self.v + (1-self.alpha) * vb

        l = (l - self.c) / max(1, self.v)
        reinforce_reward = torch.mul(l.detach(), sent_logprobs)
        baseline_error = l - self.reinforce_baseline
        baseline_term = -torch.pow(baseline_error, 2) #torch.mul(l.detach(), torch.pow(baseline_error, 2))      # Didn't work when rescaling by l as outlined in the original paper...

        estimator = reinforce_reward + baseline_term
        aux_signals = 0
        # aux_signals -= 0.012 * sent_logits.softmax(-1)[:,:,1].mean()
        # nodes_onehot = F.one_hot(nodes+1, sent_logits.size(1)+1)[:,:,1:].sum(1)
        # aux_signals += 0.05*self._compute_logprobs(nodes_onehot, sent_logits).mean()

        assert not torch.isnan(estimator).any()
        outputs = {"loss": -(estimator.mean() + aux_signals)}
 
        if True: 
            qa_output_baseline, z_bl, z_bl_onehot, batch_bl, meta_records_bl = self.benchmark(
                sent_logits=sent_logits,
                ques_logits=ques_logits,
                metadata=metadata,
                label=label,
                polarity=polarity,
                nodes=nodes,
                word_overlap_scores=word_overlap_scores,
                meta_records=meta_records,
            )

            metadata_bl = batch_bl['metadata']
            engine_labels_bl = call_theorem_prover_from_lst(instances=meta_records_bl)
            new_labels_bl = [bool(l.item()) if nl is None else nl for nl,l in zip(engine_labels_bl, label)]     # TODO: this is a hack to catch cases where problog cannot solve the logic program. Find a better solution here.
            modified_label_bl = 1 - torch.tensor(new_labels_bl, device=self._d, dtype=torch.int)

            # Verify batched sentences are same as those in new meta records
            records_sentences_bl = [[i['text'] for i in list(record['triples'].values()) + list(record['rules'].values()) if not i['mask']] for record in meta_records_bl]
            assert all([all([sent in meta['context'] for sent in sentences]) for meta, sentences in zip(metadata_bl, records_sentences_bl)])

            correct = (preds == modified_label_)
            correct_bl = (qa_output_baseline["label_probs"].argmax(-1) == modified_label_bl)
        self.log_results(qlens_, correct, ref=correct_bl)

        if True:
            sentences = [[m['question_text']] + m['context'].split('.') for m in batch['metadata']]
            records = []
            for i in range(len(correct)):
                i_ = floor(i / self._n_mc)
                record = {
                    "qa_correct": correct[i].item(),
                    "label": label[i_].item(),
                    "mod_label": modified_label_[i].item(),     # Target label (i.e. 1-correct label)
                    "polarity": polarity[i_].item(),
                    "question": sentences[i][0],
                    "orig_sentences": orig_sentences[i_],
                    "sampled_sentences": sentences[i],
                    "proof_sentences": proof_sentences[i_][1:],
                    "qa_probs": probs[i].cpu(),
                    "qa_preds": preds[i].item(),
                }
                records.append(record)
                record["label"], record["polarity"], record["mod_label"], record["qa_probs"]
            self.records.extend(records)
        return outputs

    def update_meta_sentelim(self, z, meta_records_, metadata):
        meta_records = [deepcopy(m) for m in meta_records_]
        for n, (nodes, meta) in enumerate(zip(z, metadata)):
            sentence_scramble = meta['sentence_scramble']
            n_sents = len(sentence_scramble)
            scrambled_nodes = [sentence_scramble[i] for i in nodes if i not in [-1, n_sents]]
            nfacts = len(meta_records[n]['triples'])
            nrules = len(meta_records[n]['rules'])
            qid = 'Q' + str(meta['id'].split('-')[-1])
            meta_records[n]['questions'] = {qid: meta_records[n]['questions'][qid]}
            assert meta_records[n]['questions'][qid]['question'] == meta['question_text']
            for i in range(1, nfacts+1):
                meta_records[n]['triples'][f'triple{i}']['mask'] = 0 if i in scrambled_nodes else 1
            for i in range(1, nrules+1):
                meta_records[n]['rules'][f'rule{i}']['mask'] = 0 if i + nfacts in scrambled_nodes else 1
            continue
        return meta_records, metadata

    def update_meta_quesflip(self, z, meta_records_, metadata_):
        meta_records = [deepcopy(m) for m in meta_records_]
        metadata = [deepcopy(m) for m in metadata_]
        for n, (nodes, meta) in enumerate(zip(z, metadata)):
            qid = 'Q' + meta['id'].split('-')[-1]
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
        return meta_records, metadata

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

    def log_results(self, qlens, acc, tp=None, ref=None):
        if tp is None:
            tp = torch.full_like(acc, False)
        if ref is None:
            ref = torch.full_like(acc, False)

        for d, c, r, t in zip(qlens, acc, ref.repeat_interleave(self._n_mc, dim=0), tp):
            if d not in self.answers:
                self.answers[d] = [[], [], []]
            self.answers[d][0].append(c.item())
            self.answers[d][1].append(r.item())
            if t != -1:
                self.answers[d][2].append(t)

        if self.training:
            self.print_results()

    def print_results(self):
        n = self._n_mc * 100
        for d in sorted(self.answers.keys()):
            all_score_a = self.answers[d][0].count(True) / max(len(self.answers[d][0]), 1)
            last_100_a = self.answers[d][0][-n:].count(True) / max(len(self.answers[d][0][-n:]),1)
            all_score_r = self.answers[d][1].count(True) / max(len(self.answers[d][1]),1)
            last_100_r = self.answers[d][1][-n:].count(True) / max(len(self.answers[d][1][-n:]),1)
            print(f'\nM:\tL: {d}\tAll: {all_score_a:.3f}\tLast {n}: {last_100_a:.2f}\t'
                # f'Last {n} tp: {last_100_tp:.2f}\t'
                f'B:\tAll: {all_score_r:.3f}\tLast {n}: {last_100_r:.2f}\tN: {len(self.answers[d][0])}')

    def _prep_batch(self, z, metadata, label):
        ''' Concatenate the latest retrieval to the current 
            query+retrievals. Also update the tensors for the next
            rollout pass.
        '''
        # Concatenate query + retrival to make new query_retrieval matrix of idxs
        sentences = []
        for topk, meta, e in zip(z, metadata, label):
            question = meta['question_text']
            sentence_idxs = [i for i in topk.tolist() if i != self._z_mask]
            context_rtr = [
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]
            meta['context_str'] = f"q: {question} c: {''.join(context_rtr).strip()}"
            sentences.append((question, ''.join(context_rtr).strip(), e))

        batch = self.dataset_reader.encode_batch(sentences, self.qa_vocab, disable=True)
        return self.dataset_reader.move(batch, self._d)

    def _rebatch(self, metadata, label):
        sentences = []
        for meta, e in zip(metadata, label):
            question = meta['question_text']
            context = meta['context'].split('.')[:-1]
            insert_pos = torch.randperm(len(context))[0]
            context.insert(insert_pos, ' '+'FLAG')
            context = [c + '.' for c in context]
            meta['context_str'] = f"q: {question} c: {''.join(context).strip()}"
            sentences.append((question, ''.join(context).strip(), insert_pos))

        batch = self.dataset_reader.encode_batch(sentences, self.qa_vocab)
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
        if self.training:
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

        return logits_, z, samples

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

        return logits_, z, samples

    def decode(self, i, batch):
        return self.dataset_reader.decode(batch['phrase']['tokens']['token_ids'][i]).split('</s> </s>')

    def random_benchmark(self, sent_logits=False, ques_logits=False, metadata=False, label=False, polarity=False, nodes=False, meta_records=False, **kwargs):
        with torch.no_grad():
            _, z_sent_bl, z_sent_bl_onehot = self._draw_samples(sent_logits, random_chance=True)
            _, z_ques_bl, z_ques_bl_onehot = self._draw_samples(ques_logits, random_chance=True)

            meta_records_bl = meta_records
            metadata_bl = metadata
            if 'sentence_elimination' in self.adv_perturbations:
                meta_records_bl, metadata_bl = self.update_meta_sentelim(z_sent_bl, meta_records_bl, metadata_bl)
            if 'question_flip' in self.adv_perturbations:
                meta_records_bl, metadata_bl = self.update_meta_quesflip(z_ques_bl, meta_records_bl, metadata_bl)

            batch_bl = self._prep_batch(z_sent_bl, metadata_bl, label)
            qa_output_baseline = self.qa_model(**batch_bl)

        return qa_output_baseline, z_sent_bl, z_sent_bl_onehot, batch_bl, meta_records_bl

    def wordscore_benchmark(self, sent_logits=False, metadata=False, label=False, polarity=False, nodes=False, word_overlap_scores=False, **kwargs):
        with torch.no_grad():
            logits_baseline, z_baseline, z_bl_onehot = self._draw_samples_wordscore(sent_logits, word_overlap_scores, random_chance=True)
            batch_baseline = self._prep_batch(z_baseline, metadata, label)
            qa_output_baseline = self.qa_model(**batch_baseline)

        return qa_output_baseline, z_baseline, z_bl_onehot, batch_baseline

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        answers = [v for value in self.answers.values() for v in value[0]]      # Flatten lists for main model
        if reset == True and not self.training:
            return {
                'EM': answers.count(True)/len(answers),
                'predictions': None,
            }
        else:
            return {
                'EM': answers.count(True)/len(answers),
            }
    
    def reset(self, dset, epoch):
        if len(self.answers) > 0:
            self.prev_answers[dset].append(self.answers)
            print(f'\n\n{dset} results for epoch {epoch}: \n\n')
            self.print_results()
        self.answers = {}


class _BaseSentenceClassifier(Model):
    def __init__(self, variant, vocab, dataset_reader, has_naf, regularizer=None, num_labels=1):
        super().__init__(vocab, regularizer)
        self._predictions = []

        self.variant = variant
        self.dataset_reader = dataset_reader
        self.model = AutoModel.from_pretrained(variant)
        assert 'roberta' in variant     # Only implemented for roberta currently

        transformer_config = self.model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self.model.config.hidden_size

        # unifing all model classification layer
        self.node_class_method = 'mean'     # ['mean', 'concat_first_last']
        self.node_class_k = 1 #if self.node_class_method == 'mean' else 2
        self.node_classifier = NodeClassificationHead(self._output_dim * self.node_class_k, 0, num_labels)
        self.naf_layer = nn.Linear(self._output_dim, self._output_dim)
        self.has_naf = has_naf

        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()
        self.num_labels = num_labels

        # Split sentences in the context based on full stop
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')

        self._p0 = torch.tensor(-float(1e9))

        self.e_true = torch.tensor(
            [self.dataset_reader.encode_token(tok, mode='retriever') for tok in '<s> ĠE : ĠTrue </s>'.split()]
        )
        self.e_false = torch.tensor(
            [self.dataset_reader.encode_token(tok, mode='retriever') for tok in '<s> ĠE : ĠFalse </s>'.split()]
        )

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
        node_reprs = torch.full((x.size(0), max_num_sentences+1, self.num_labels), self._p0).to(self._d)      # shape: (bsz, # sentences, 2)
        for b in range(x.size(0)):
            # Create list of end idxs of each context item
            end_idxs = (x[b] == self.split_idx).nonzero().squeeze(1).tolist()
            q_end = end_idxs[0]
            q_start = 2             # Beginning of question ix. Ignores BOS tokens
            start_idxs = [q_start, q_end + 4] + end_idxs[1:-1]       # +4 because tok adds four "decorative" tokens at the beginning of the context
            offsets = list(zip(start_idxs, end_idxs))

            # Form tensor containing embedding of first and last token for each sentence
            n_sentences = len(offsets)
            reprs = torch.zeros(n_sentences, self.node_class_k, embs.size(-1)).to(self._d)      # shape: (# context items, 2, model_dim)
            for i in range(n_sentences):
                start_idx = offsets[i][0] + 1            # +1 to skip full stop at beginning
                end_idx = offsets[i][1] + 1             # +1 to include full stop at end
                
                # Extract reprs for tokens in the sentence from the original encoded sequence
                reprs[i, 0] = embs[b, start_idx:end_idx].mean(dim=0)

            # Pass through classifier
            reprs_ = torch.cat((reprs.view(reprs.size(0), -1), naf_repr[b].unsqueeze(0)), 0)
            node_logits = self.node_classifier(reprs_).squeeze(-1)                
            node_reprs[b, :len(node_logits)] = node_logits

        sent_reprs = node_reprs[:,1:,:]
        ques_reprs = node_reprs[:,:1,:]

        return ques_reprs, sent_reprs


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

        return node_reprs, _


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


def gs(logits, tau=1):
    ''' Sample using Gumbel Softmax. Ingests raw logits.
    '''
    return F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)


# self.dataset_reader.decode(phrase['tokens']['token_ids'][0][10:19]).split('</s> </s>')
