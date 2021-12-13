from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
import wandb
import random

import torch
from torch.nn.modules.linear import Linear
from torch import nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)
from .utils import safe_log, right_pad, batch_lookup, EPSILON, make_dot, set_dropout, one_hot, lmap, lfilter
from .transformer_binary_qa_model import TransformerBinaryQA
# from .baseline import Baseline

torch.manual_seed(0)

@Model.register("gumbel_softmax_unified")
class GumbelSoftmaxRetrieverReasoner(Model):
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
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.qa_model = qa_model
        self.qa_model._loss = nn.CrossEntropyLoss(reduction='none')
        self.qa_vocab = qa_model.vocab
        self.vocab = vocab
        self.dataset_reader = dataset_reader
        self.regularizer = regularizer
        self.variant = variant
        self.sentence_embedding_method = sentence_embedding_method
        self.similarity_func = 'inner' #'linear' #'inner'       # TODO: set properly

        # Rollout params
        self.x = -111
        self.gamma = 1          # TODO
        self.beta = 1           # TODO
        self.num_rollout_steps = topk
        self.retriever_model = None
        self.run_analysis = False   # TODO
        # self.baseline = 0
        self.training = True
        self._context_embs = None
        self.b = None
        self._flag = False
        self._replay_memory = None
        self._mode = 'retrieval'
        self.b = 0.0
        # self.b = Baseline()

        self.define_modules()
        self.answers = {}

    def forward(self, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        **kwargs,
    ) -> torch.Tensor:
        ''' Forward pass of the network. 
            Details tbc.
        '''
        _qr = retrieval['tokens']['token_ids']
        qr = _qr[:]       # shape = (bsz, context_len, sentence_len)
        _d = qr.device
        
        # # Helper code
        flag = False
        qlens = [m['QLen'] for m in metadata]
        # self.ms = torch.tensor([m['exact_match'] for m in metadata]).to(_d)
        # nl = [torch.tensor(m['node_label'][:-1]).nonzero().squeeze().to(_d) for m in metadata] # [:-1] because final node is NAF node
        # naf = [torch.tensor(m['node_label'][-1:]).nonzero().squeeze().to(_d) for m in metadata]
        # i = 0
        # ids = [m['id'] for m in metadata]
        # # print(ids)

        # Storage tensors
        policies, actions, unscaled_retrieval_losses = [],[],[]
        
        # Retrieval rollout phase
        for t in range(self.num_rollout_steps):
            policy = self.get_retrieval_distr(qr, metadata)            
            action = self.gs(policy, tau=1) if self.training else one_hot(policy, policy.argmax(-1))
            if flag:
                action = one_hot(action, torch.tensor([0]*action.size(0)).view(-1,1).cuda())
            loss = self.retriever_loss(policy, action.argmax(-1))

            policies.append(policy)
            actions.append(action)
            unscaled_retrieval_losses.append(loss)

            q = qr.gather(1, action.argmax(-1).view(-1, 1, 1).repeat(1, 1, qr.size(-1))).squeeze(1)
            if t == self.num_rollout_steps:
                metadata = self.prep_next_batch(qr, metadata, actions, t, False)
            else:
                qr, metadata = self.prep_next_batch(qr, metadata, actions, t, True)


            if True:
                a = action.argmax(-1)
                p = policy.argmax(-1)
                p_ = policy.softmax(-1)
                argmax_ps = p_.gather(1, p.unsqueeze(1)).squeeze()
                action_ps = p_.gather(1, a.unsqueeze(1)).squeeze()
            
        # Query answering phase
        self.update_meta(q, metadata, actions)
        output = self.answer(q, label, metadata)

        # Scale retrieval losses by final loss
        qa_loss = output['loss'].detach()
        qa_scale = torch.gather(output['label_probs'].detach(), dim=1, index=label.unsqueeze(1))
        unscaled_retrieval_losses_ = torch.cat([u.unsqueeze(1) for u in unscaled_retrieval_losses], dim=1)
        retrieval_losses = (qa_scale - self.b()) * unscaled_retrieval_losses_ / unscaled_retrieval_losses_.size(1)      # NOTE: originals
        # retrieval_losses = (qa_loss.unsqueeze(1) - self.b) * unscaled_retrieval_losses_ / unscaled_retrieval_losses_.size(1)      # NOTE: originals
        # total_loss = qa_loss + unscaled_retrieval_losses_ #retrieval_losses
        total_loss = retrieval_losses
        output['loss'] = total_loss.mean()
        # output['loss'] = loss
        # output['loss'] = unscaled_retrieval_losses_.mean()

        # Record trajectory data
        output['unnorm_policies'] = policies
        output['sampled_actions'] = torch.cat([a.unsqueeze(0) for a in actions]).argmax(dim=-1)

        correct = (output["label_probs"].argmax(-1) == label)
        self.log_results(qlens, correct)
        print(f'\n\n{qa_loss.mean().item():.3f}\t{retrieval_losses.mean().item():.3f}\t{(correct).float().mean().item()}')

        if self._replay_memory is not None:
            self.add_correct_to_buffer(correct, metadata)

        return output

    def log_results(self, qlens, correct):
        for d, c in zip(qlens, correct):
            if d not in self.answers:
                self.answers[d] = []
            self.answers[d].append(c.item())

        for d in self.answers.keys():
            all_score = self.answers[d].count(True) / len(self.answers[d])
            last_100 = self.answers[d][-100:].count(True) / len(self.answers[d][-100:])
            print(f'\nL: {d}\tAll: {all_score:.4f}\tLast 100: {last_100:.2f}\tN: {len(self.answers[d])}')

    def get_retrieval_distr(self, qr, meta=None):
        ''' Compute the probability of retrieving each item given
            the current query+retrieval (i.e. p(zj | zi, y))
        '''
        e_q = self.get_context_embs(qr)
        sim = self.W(e_q).squeeze(-1)

        # Ensure padding receives 0 probability mass
        similarity = torch.where(
            qr.max(dim=2).values == self.retriever_pad_idx,     # Identify rows which contain all padding
            torch.tensor(-float("inf")).to(e_q.device), 
            sim,
        )

        # Deal with nans- these are caused by all sentences being padding.
        similarity[torch.isinf(similarity).all(dim=-1)] = 1 / similarity.size(0)
        if torch.isinf(similarity).all(dim=-1).any():
            raise ValueError('All retrievals are -inf for a sample. This will lead to nan loss')

        return similarity
        
    def answer(self, qr, label, metadata):
        return self.get_query_embs(qr, label, metadata)
    
    def get_query_embs(self, qr, label=None, metadata=None):
        qr_ = {'tokens': {'token_ids': qr, 'type_ids': torch.zeros_like(qr)}}
        return self.qa_model(qr_, label, metadata)

    def get_context_embs(self, c):
        return self.retriever_model(c)

    def gs(self, logits, tau=1):
        ''' Gumbel softmax
        '''
        return F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)

    def update_meta(self, query_retrieval, metadata, actions):
        ''' Log relevant metadata for later use.
        '''
        retrievals = torch.cat([a.unsqueeze(0) for a in actions]).argmax(dim=-1).T
        for qr, topk, meta in zip(query_retrieval, retrievals, metadata):
            meta['topk'] = topk.tolist()
            meta['query_retrieval'] = qr.tolist()

    def prep_next_batch(self, qr, metadata, actions, t, return_qr):
        ''' Concatenate the latest retrieval to the current 
            query+retrievals. Also update the tensors for the next
            rollout pass.
        '''
        # Get indexes of retrieval items
        retrievals = torch.cat([a.unsqueeze(0) for a in actions]).argmax(-1).T

        # Concatenate query + retrival to make new query_retrieval matrix of idxs        
        sentences = []
        for topk, meta in zip(retrievals, metadata):
            question = meta['question_text']
            sentence_idxs = [int(i) for i in topk.tolist()[:t+1] if i != self.x]
            context_rtr = [
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]
            meta['context_str'] = f"q: {question} c: {''.join(context_rtr).strip()}"
            sentences.append((question, ''.join(context_rtr).strip(), meta['context']))

        if not return_qr:
            return metadata

        batch = self.dataset_reader.transformer_indices_from_qa(sentences, self.qa_vocab)
        qr_ = batch['retrieval']['tokens']['token_ids'].to(qr.device)
        qr_ = qr_.scatter(
            1, retrievals.unsqueeze(-1).repeat(1, 1, qr_.size(-1)), self.retriever_pad_idx
        )
        return qr_, metadata

    def baseline_loss(self, b, x):
        # return -torch.zeros_like(x).fill_(-torch.log(torch.tensor(1-b)))
        return -torch.zeros_like(x).fill_(torch.log(torch.tensor(b)))

    def define_modules(self):
        if self.variant == 'spacy':
            self.retriever_model = SpacyRetrievalEmbedder(
                sentence_embedding_method=self.sentence_embedding_method,
                vocab=self.vocab,
                variant=self.variant,
            )
            self.tok_name = 'tokens'
            self.retriever_pad_idx = self.vocab.get_token_index(self.vocab._padding_token)      # TODO: standardize these
        elif 'roberta' in self.variant:
            self.retriever_model = TransformerRetrievalEmbedder(
                sentence_embedding_method=self.sentence_embedding_method,
                vocab=self.vocab,
                variant=self.variant,
            )
            self.tok_name = 'token_ids'
            self.retriever_pad_idx = self.dataset_reader.pad_idx(mode='retriever')       # TODO: standardize these
        else:
            raise ValueError(
                f"Invalid retriever_variant: {self.variant}.\nInvestigate!"
            )

        if self.similarity_func == 'linear':
            self.proj = nn.Linear(2*self.qa_model._output_dim, 1)      # TODO: sort for different retriever and qa models

        self.retriever_loss = nn.CrossEntropyLoss(reduction='none')
        self.W = nn.Linear(self.retriever_model.embedder.config.hidden_size, 1)        # TODO

        set_dropout(self.retriever_model, 0.0)
        # set_dropout(self.qa_model, 0.0)

    def set_mode(self, mode: str):
        assert mode in ['binary_classification', 'retrieval']
        self._mode = mode
        
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        if self._mode == 'retrieval':
            return self.qa_model.get_metrics(reset=reset)
        elif self._mode == 'binary_classification':
            return self.retriever_model.get_metrics(reset=reset)
        else:
            raise NotImplementedError


@Model.register("gumbel_softmax_pg")
class ProgressiveDeepeningGumbelSoftmaxRetsrieverReasoner(GumbelSoftmaxRetrieverReasoner):
    def __init__(self,
        qa_model: Model,
        variant: str,
        vocab: Vocabulary = None,
        requires_grad: bool = True,
        transformer_weights_model: str = None,
        num_labels: int = 2,
        predictions_file = None,
        layer_freeze_regexes: List[str] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        topk: int = 5,
        sentence_embedding_method: str = 'mean',
        dataset_reader = None,
        mode = 'retrieval',
    ) -> None:
        super().__init__(
            qa_model,
            variant,
            vocab,
            requires_grad,
            transformer_weights_model,
            num_labels,
            predictions_file,
            layer_freeze_regexes,
            regularizer,
            topk,
            sentence_embedding_method,
            dataset_reader,
        )
        self._mode = mode
        self._state = True

    def forward_retreival(self, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        **kwargs,
    ) -> torch.Tensor:
        ''' Forward pass of the network. 
            Details tbc.
        '''
        _qr = retrieval['tokens']['token_ids']
        qr = _qr[:]       # shape = (bsz, context_len, sentence_len)
        _d = qr.device
        
        # # Helper code
        qlens = [m['QLen'] for m in metadata]

        # Storage tensors
        policies, actions, unscaled_retrieval_losses = [],[],[]
        
        # Retrieval rollout phase
        for t in range(self.num_rollout_steps):
            policy = self.get_retrieval_distr(qr, metadata)            
            action = self.gs(policy, tau=1) if self.training else one_hot(policy, policy.argmax(-1))
            loss = self.retriever_loss(policy, action.argmax(-1))

            policies.append(policy)
            actions.append(action)
            unscaled_retrieval_losses.append(loss)

            q = qr.gather(1, action.argmax(-1).view(-1, 1, 1).repeat(1, 1, qr.size(-1))).squeeze(1)
            if t == self.num_rollout_steps:
                metadata = self.prep_next_batch(qr, metadata, actions, t, False)
            else:
                qr, metadata = self.prep_next_batch(qr, metadata, actions, t, True)

            if True:
                a = action.argmax(-1)
                p = policy.argmax(-1)
                p_ = policy.softmax(-1)
                argmax_ps = p_.gather(1, p.unsqueeze(1)).squeeze()
                action_ps = p_.gather(1, a.unsqueeze(1)).squeeze()
            
        # Query answering phase
        self.update_meta(q, metadata, actions)
        output = self.answer(q, label, metadata)

        # Scale retrieval losses by qa output
        qa_loss = output['loss'].detach()
        qa_scale = torch.gather(output['label_probs'].detach(), dim=1, index=label.unsqueeze(1))
        unscaled_retrieval_losses_ = torch.cat([u.unsqueeze(1) for u in unscaled_retrieval_losses], dim=1)
        retrieval_losses = (qa_scale - self.b) * unscaled_retrieval_losses_ / unscaled_retrieval_losses_.size(1)      # NOTE: originals
        total_loss = retrieval_losses
        output['loss'] = total_loss.mean()

        # Record trajectory data
        output['unnorm_policies'] = policies
        output['sampled_actions'] = torch.cat([a.unsqueeze(0) for a in actions]).argmax(dim=-1)

        correct = (output["label_probs"].argmax(-1) == label)
        self.log_results(qlens, correct)
        print(f'\n\n{qa_loss.mean().item():.3f}\t{retrieval_losses.mean().item():.3f}\t{(correct).float().mean().item()}')

        self.add_correct_to_buffer(correct, metadata) #[mem for mem in self._replay_memory.memory if mem['QLen'] == 2][0]

        return output

    def forward_binclass(self, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        **kwargs,
    ) -> torch.Tensor:
        self._d = label.device
        query, constructed_labels = self.prep_batch(metadata)
        batch = {
            'phrase': {'tokens': {'token_ids': query, 'type_ids': torch.zeros_like(query)}}, 
            'label': constructed_labels, 
            'metadata': metadata
        }
        output = self.retriever_model(**batch)
        return output

    def __call__(self, *args, **kwargs):
        ''' Redefine the forward pass depending on what task this
            model is performing.
        '''
        if self._mode == 'retrieval':
            return self.forward_retreival(*args, **kwargs)
        elif self._mode == 'binary_classification':
            return self.forward_binclass(*args, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        ''' To catch references to "forward" for this class.
        '''
        return self(*args, **kwargs)

    def add_correct_to_buffer(self, outcomes, metadata):
        ''' Add the correctly answered questions to the replay buffer
        '''
        for outcome, meta in zip(outcomes, metadata):
            if outcome and meta['QLen'] == self.num_rollout_steps:
                self._replay_memory.push(meta)

    def prep_batch(self, metadata):
        ''' Constructs a positive or negative binary classification
            sample given from the metadata
        '''
        # Concatenate query + retrival to make new query_retrieval matrix of idxs        
        sentences, labels = [], []
        for meta in metadata:
            question = meta['question_text']
            retrievals = meta['psuedolabel_retrievals']
            nodes = meta['node_label']
            context = meta['context']
            support = retrievals[:-1]
            if self._state:
                label_idx = retrievals[-1:]
                labels.append(1)
            else:
                label_idx = random.sample([x for x in range(len(nodes[:-1])) if x not in retrievals], 1)
                labels.append(0)
                
            sentence_idxs = support + label_idx
            # context_rtr = [
            #     toks + '.' for n, toks in enumerate(context.split('.')[:-1]) 
            #     if n in sentence_idxs
            # ]
            strs = context.split('.')[:-1]
            context_rtr = [strs[i] + '.' for i in sentence_idxs]
            meta['context_str'] = f"q: {question} c: {''.join(context_rtr).strip()}"
            assert context_rtr != []
            sentences.append((question, '', ''.join(context_rtr).strip()))

            self._state = not self._state

        batch = self.dataset_reader.transformer_indices_from_qa(sentences, self.qa_vocab)
        query = batch['phrase']['tokens']['token_ids'].to(self._d)
        labels_ = torch.tensor(labels).to(self._d)

        return query, labels_

    def get_context_embs(self, c):
        # return self.retriever_model(c)
        c_ = {'tokens': {'token_ids': c, 'type_ids': torch.zeros_like(c)}}
        output = self.retriever_model(c_)
        return output['label_logits'].view(c.size(0), c.size(1), -1)

    def define_modules(self):
        self.retriever_model = TransformerBinaryQA(vocab=self.vocab, pretrained_model=self.variant, num_labels=1)
        self.tok_name = 'token_ids'
        self.retriever_pad_idx = self.dataset_reader.pad_idx(mode='retriever')       # TODO: standardize these

        if self.similarity_func == 'linear':
            self.proj = nn.Linear(2*self.qa_model._output_dim, 1)      # TODO: sort for different retriever and qa models

        self.retriever_loss = nn.CrossEntropyLoss(reduction='none')

        # self.W = nn.Linear(2, 1)

        set_dropout(self.retriever_model, 0.0)
        # set_dropout(self.qa_model, 0.0)

    def get_retrieval_distr(self, qr, meta=None):
        ''' Compute the probability of retrieving each item given
            the current query+retrieval (i.e. p(zj | zi, y))
        '''
        e_q = self.get_context_embs(qr).squeeze(-1)
        # e_q = e_q[...,0]
        # e_q = self.W(e_q).squeeze(-1)
        # e_q = e_q.max(-1).values

        # Ensure padding receives 0 probability mass
        similarity = torch.where(
            qr.max(dim=2).values == self.retriever_pad_idx,     # Identify rows which contain all padding
            torch.tensor(-float("inf")).to(e_q.device), 
            e_q,
        )

        # Deal with nans- these are caused by all sentences being padding.
        similarity[torch.isinf(similarity).all(dim=-1)] = 1 / similarity.size(0)
        if torch.isinf(similarity).all(dim=-1).any():
            raise ValueError('All retrievals are -inf for a sample. This will lead to nan loss')

        return similarity

    def decode(self, ids):
        if ids.ndim == 2:
            return lmap(self.dataset_reader.decode, ids)
        elif ids.ndim == 1:
            return self.dataset_reader.decode(ids)
        else:
            raise NotImplementedError
