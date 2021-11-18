from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
from math import floor

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
# from .model import RobertaForPRVI

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
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.variant = variant
        self.qa_model = qa_model        # TODO: replace with fresh transformerbinaryqa
        self.qa_model._loss = nn.CrossEntropyLoss(reduction='none')
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self._mlloss = nn.BCEWithLogitsLoss(reduction='none')
        self.qa_vocab = qa_model.vocab
        self.dataset_reader = dataset_reader
        # self.infr_model = InferenceNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, has_naf=add_NAF, num_labels=1)
        self.gen_model = GenerativeNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, has_naf=add_NAF, num_labels=2)
        self.vocab = vocab
        self.regularizer = regularizer
        self.sentence_embedding_method = sentence_embedding_method
        self.num_labels = num_labels
        self.objective = objective
        # self.objective_fn = getattr(self, objective.lower())
        
        self._n_z = topk        # Number of latent retrievals
        self._beta = 0.1        # Scale factor for KL_div TODO: set dynamically
        self._p0 = torch.tensor(-float(1e9))        # TODO: set dynamically
        self._n_mc = num_monte_carlo          # Number of monte carlo steps
        self._logprob_method = 'CE'      # TODO: set dynamically
        self._z_mask = -1
        self._do_mask_z = do_mask_z     # TODO: set dynamically
        self._additional_qa_training = additional_qa_training
        self.threshold_sampling = threshold_sampling

        self._baseline_type = baseline_type
        # if baseline_type == 'Prob-NMN':
        #     self._reinforce = Reinforce(baseline_decay=0.99)
        # elif baseline_type == 'NVIL':
        #     self.baseline_model = BaselineNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader, num_labels=1)
        #     self._reinforce = TrainableReinforce(self.baseline_model)
        #     set_dropout(self.baseline_model, 0.0)

        self.answers = {}
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
        gen_logits = self.gen_model(phrase)

        # Take multiple monte carlo samples by tiling the logits
        gen_logits_ = gen_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)
        polarity_ = polarity.repeat_interleave(self._n_mc, dim=0)
        nodes_ = nodes.repeat_interleave(self._n_mc, dim=0)

        logits, z, z_onehot = self._draw_samples(gen_logits)        # (bsz * mc steps, num retrievals)


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
        gen_logprobs = self._compute_logprobs(z_onehot, gen_logits_)

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
        # reinforce_reward = torch.mul(l.detach(), gen_logprobs)
        # baseline_error = l - self.reinforce_baseline
        # baseline_term = -torch.pow(baseline_error, 2) #torch.mul(l.detach(), torch.pow(baseline_error, 2))      # Didn't work when rescaling by l as outlined in the original paper...

        # estimator = reinforce_reward + baseline_term
        aux_signals = 0
        # aux_signals -= 0.012 * gen_logits.softmax(-1)[:,:,1].mean()
        nodes_onehot = F.one_hot(nodes+1, gen_logits.size(1)+1)[:,:,1:].sum(1)
        # aux_signals += 0.1*self._compute_logprobs(nodes_onehot, gen_logits).mean()
        aux_signals += self._compute_logprobs(nodes_onehot, gen_logits).mean()

        # assert not torch.isnan(estimator).any()
        # outputs = {"loss": -(estimator.mean() + aux_signals)}
        outputs = {"loss": -(aux_signals)}
 
        if True:
            # sentences = [[m['question_text']] + m['context'].split('.') for m in batch['metadata']]
            # correct = (preds == modified_label_)
            correct = torch.tensor([set(i.item() for i in n if i!=-1) == set(i.item() for i in z_ if i!=-1) for n,z_ in zip(nodes_,z)])
        self.log_results(qlens_, correct)

        return outputs

    def forward_(self, phrase=None, label=None, metadata=None, retrieval=None, **kwargs) -> torch.Tensor:
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
        gen_logits = self.gen_model(phrase)

        # Take multiple monte carlo samples by tiling the logits
        gen_logits_ = gen_logits.repeat_interleave(self._n_mc, dim=0)
        label_ = label.repeat_interleave(self._n_mc, dim=0)
        polarity_ = polarity.repeat_interleave(self._n_mc, dim=0)
        nodes_ = nodes.repeat_interleave(self._n_mc, dim=0)

        logits, z, z_onehot = self._draw_samples(gen_logits)        # (bsz * mc steps, num retrievals)


        # Modify labels to reflect adversarial goals
        # Stick with label if polarity == label and proof nodes not in sampled nodes
        # Else flip label
        sampled_proof_nodes = torch.tensor([all([n in _z for n in node if n!=-1]) for _z,node in zip(z, nodes_)], device=self._d)
        flip_mask = 1 - (polarity_ == label_).int() * (1-sampled_proof_nodes.int())
        modified_label_ = (label_ != flip_mask).long()

        batch = self._prep_batch(z, metadata_, modified_label_)
        with torch.no_grad():
            qa_output = self.qa_model(**batch)

        qa_logprobs = -qa_output['loss']
        qa_probs = qa_logprobs.exp()
        probs = qa_output["label_probs"]
        preds = probs.argmax(-1)
        qa_logits = qa_output['label_logits']
        qa_logprobs = -self.qa_model._loss(qa_logits, modified_label_)
        # qa_logprobs = torch.where(torch.isnan(qa_logprobs), torch.tensor(0., device=self._d), qa_logprobs)
        gen_logprobs = self._compute_logprobs(z_onehot, gen_logits_)

        # Compute objective function
        # l = qa_logprobs.detach()      # Using Jensen's inquaity
        l = qa_probs.detach()           # Without Jensen's inquaity
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
        # aux_signals -= 0.012 * gen_logits.softmax(-1)[:,:,1].mean()
        nodes_onehot = F.one_hot(nodes+1, gen_logits.size(1)+1)[:,:,1:].sum(1)
        aux_signals += 0.1*self._compute_logprobs(nodes_onehot, gen_logits).mean()

        assert not torch.isnan(estimator).any()
        outputs = {"loss": -(estimator.mean() + aux_signals)}

        # with torch.no_grad():
        #     logits_baseline, z_baseline = self._draw_samples(gen_logits, random_chance=True)
        #     batch_baseline = self._prep_batch(z_baseline, metadata, label)
        #     qa_output_baseline = self.qa_model(**batch_baseline)
        #     correct_baseline = (qa_output_baseline["label_probs"].argmax(-1) == label)
 
        if True:
            sentences = [[m['question_text']] + m['context'].split('.') for m in batch['metadata']]
            correct = (preds == modified_label_)
        self.log_results(qlens_, correct)

        if True:
            for i in range(len(correct)):
                i_ = floor(i / self._n_mc)
                record = {
                    "qa_correct": correct[i].item(),
                    "label": label[i_].item(),
                    "mod_label": modified_label_[i].item(),
                    "polarity": polarity[i_].item(),
                    "question": sentences[i][0],
                    "orig_sentences": orig_sentences[i_],
                    "sampled_sentences": sentences[i],
                    "proof_sentences": proof_sentences[i_][1:],
                    "qa_probs": probs[i].cpu(),
                    "qa_preds": preds[i].item(),
                }
                self.records.append(record)
                record["label"], record["polarity"], record["mod_label"], record["qa_probs"]

        return outputs

    def forward_basicsanitycheck(self, phrase=None, label=None, metadata=None, retrieval=None, **kwargs) -> torch.Tensor:
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

        # self.dataset_reader.encode_token('FLAG', 'retriever')
        batch = self._rebatch(metadata, label)
        phrase = batch['phrase']
        metadata = batch['metadata']
        metadata_ = [m for m in metadata for _ in range(self._n_mc)]
        orig_sentences = [[m['question_text']] + m['context'].split('.') for m in metadata]

        # nodes = [torch.tensor(m['node_label'][:-1]).nonzero().squeeze(1) for m in metadata]
        # max_nodes = max(len(n) for n in nodes)
        # nodes = torch.tensor([n.tolist() + [-1]*(max_nodes-len(n)) for n in nodes], device=self._d)
        nodes = batch['label'].unsqueeze(1)
        proof_sentences = [[s[0]]+[s[i+1] for i in n if i!=-1] for n,s in zip(nodes, orig_sentences)]

        # Obtain retrieval logits
        gen_logits = self.gen_model(phrase)

        # Take multiple monte carlo samples by tiling the logits
        gen_logits_ = gen_logits.repeat_interleave(self._n_mc, dim=0)
        # label_ = label.repeat_interleave(self._n_mc, dim=0)
        polarity_ = polarity.repeat_interleave(self._n_mc, dim=0)
        nodes_ = nodes.repeat_interleave(self._n_mc, dim=0)

        logits, z, z_onehot = self._draw_samples(gen_logits)        # (bsz * mc steps, num retrievals)


        # Modify labels to reflect adversarial goals
        # Stick with label if polarity == label and proof nodes not in sampled nodes
        # Else flip label
        # sampled_proof_nodes = torch.tensor([all([n in _z for n in node if n!=-1]) for _z,node in zip(z, nodes_)], device=self._d)
        # flip_mask = 1 - (polarity_ == label_).int() * (1-sampled_proof_nodes.int())
        # modified_label_ = (label_ != flip_mask).long()
        modified_label_ = (nodes == z).any(-1).float()
        modified_label_ -= 0.1*(modified_label_ - (modified_label_ == 0).float())

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
        gen_logprobs = self._compute_logprobs(z_onehot, gen_logits_)

        qa_probs = modified_label_
        # Compute objective function
        # l = qa_logprobs.detach()      # Using Jensen's inquaity
        l = qa_probs.detach()           # Without Jensen's inquaity
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
        aux_signals -= 0.012 * gen_logits.softmax(-1)[:,:,1].mean()
        # nodes_onehot = F.one_hot(nodes+1, gen_logits.size(1)+1)[:,:,1:].sum(1)
        # aux_signals += 0.1*self._compute_logprobs(nodes_onehot, gen_logits).mean()

        assert not torch.isnan(estimator).any()
        outputs = {"loss": -(estimator.mean() + aux_signals)}

        # with torch.no_grad():
        #     logits_baseline, z_baseline = self._draw_samples(gen_logits, random_chance=True)
        #     batch_baseline = self._prep_batch(z_baseline, metadata, label)
        #     qa_output_baseline = self.qa_model(**batch_baseline)
        #     correct_baseline = (qa_output_baseline["label_probs"].argmax(-1) == label)
 
        if True:
            # sentences = [[m['question_text']] + m['context'].split('.') for m in batch['metadata']]
            # correct = (preds == modified_label_)
            correct = (nodes == z).any(-1)
        self.log_results(qlens_, correct)

        if False:
            for i in range(len(correct)):
                i_ = floor(i / self._n_mc)
                record = {
                    "qa_correct": correct[i].item(),
                    "label": label[i_].item(),
                    "mod_label": modified_label_[i].item(),
                    "polarity": polarity[i_].item(),
                    "question": sentences[i][0],
                    "orig_sentences": orig_sentences[i_],
                    "sampled_sentences": sentences[i],
                    "proof_sentences": proof_sentences[i_][1:],
                    "qa_probs": probs[i].cpu(),
                    "qa_preds": preds[i].item(),
                }
                self.records.append(record)
                record["label"], record["polarity"], record["mod_label"], record["qa_probs"]

        return outputs

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

        n = self._n_mc * 100
        for d, c, r, t in zip(qlens, acc, ref.repeat_interleave(self._n_mc, dim=0), tp):
            if d not in self.answers:
                self.answers[d] = [[], [], []]
            self.answers[d][0].append(c.item())
            self.answers[d][1].append(r.item())
            if t != -1:
                self.answers[d][2].append(t)

        for d in sorted(self.answers.keys()):
            all_score_a = self.answers[d][0].count(True) / max(len(self.answers[d][0]), 1)
            last_100_a = self.answers[d][0][-n:].count(True) / max(len(self.answers[d][0][-n:]),1)
            last_100_tp = self.answers[d][2][-n:].count(True) / max(len(self.answers[d][2][-n:]),1)
            all_score_r = self.answers[d][1].count(True) / max(len(self.answers[d][1]),1)
            last_100_r = self.answers[d][1][-n:].count(True) / max(len(self.answers[d][1][-n:]),1)
            print(f'\nM:\tL: {d}\tAll: {all_score_a:.3f}\tLast {n}: {last_100_a:.2f}\t')
            # print(f'\nM:\tL: {d}\tAll: {all_score_a:.3f}\tLast {n}: {last_100_a:.2f}\t'
            #     f'Last {n} tp: {last_100_tp:.2f}\t'
            #     f'B:\tAll: {all_score_r:.3f}\tLast {n}: {last_100_r:.2f}\tN: {len(self.answers[d][0])}')
            
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

        batch = self.dataset_reader.encode_batch(sentences, self.qa_vocab)
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
        samples = gs(logits_).argmax(-1)
        mask = (logits_ == -1e9).all(-1).int()
        samples *= 1 - mask       # Ensure padding sentences are not sampled
        samples = samples.view(logits.shape[:-1])

        max_draws = samples.sum(-1).max()
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

        return logits_, z, samples

    def decode(self, i, batch):
        return self.dataset_reader.decode(batch['phrase']['tokens']['token_ids'][i]).split('</s> </s>')


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

        max_num_sentences = (x == self.split_idx).nonzero()[:,0].bincount().max() - int(self.has_naf)
        node_reprs = torch.full((x.size(0), max_num_sentences, self.num_labels), self._p0).to(self._d)      # shape: (bsz, # sentences)
        for b in range(x.size(0)):
            # Create list of end idxs of each context item
            end_idxs = (x[b] == self.split_idx).nonzero().squeeze().tolist()
            q_end = end_idxs.pop(0)     # Remove first item as this is the question
            end_idxs.insert(0, q_end + 4)   # Tokenizer adds four "decorative" tokens at the beginning of the context

            # Form tensor containing embedding of first and last token for each sentence
            n_sentences = len(end_idxs) - 1 - int(self.has_naf)
            reprs = torch.zeros(n_sentences, self.node_class_k, embs.size(-1)).to(self._d)      # shape: (# context items, 2, model_dim)
            for i in range(n_sentences):
                start_idx = end_idxs[i] + 1            # +1 to skip full stop at beginning
                end_idx = end_idxs[i+1] + 1            # +1 to include full stop at end
                
                # Extract reprs for tokens in the sentence from the original encoded sequence
                reprs[i, 0] = embs[b, start_idx:end_idx].mean(dim=0)

            # Pass through classifier
            reprs_ = torch.cat((reprs.view(reprs.size(0), -1), naf_repr[b].unsqueeze(0)), 0)
            node_logits = self.node_classifier(reprs_).squeeze(-1)                
            node_reprs[b, :len(node_logits)] = node_logits

        return node_reprs

    def _encode_and_append_label(self, encoded, label):
        ''' Tokenizes and appends the label to the already 
            encoded context
        '''
        len_e = len(self.e_true)
        eqc = torch.zeros(encoded.size(0), encoded.size(1) + len_e, dtype=torch.long).to(self._d)
        eqc[:, len_e:] = encoded

        # Add the encoded version of the label "<s> ĠE: [ĠTrue/ĠFalse] </s>"
        for b in range(label.size(0)):
            if label[b] == 1:
                eqc[b,:len_e] = self.e_true.to(self._d)
            elif label[b] == 0:
                eqc[b,:len_e] = self.e_false.to(self._d)
            else:
                raise ValueError
                
        return eqc


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


class Reinforce(nn.Module):
    """ TODO: from probnmn
    """
    def __init__(self, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce_baseline = 0.0
        self._baseline_decay = baseline_decay

    def forward(self, inputs, reward, *args):
        # Detach the reward term, we don't want gradients to flow to through it.
        centered_reward = reward.detach() - self._reinforce_baseline

        # Update moving average baseline.
        self._reinforce_baseline += self._baseline_decay * centered_reward.mean().item()
        return inputs * centered_reward


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
