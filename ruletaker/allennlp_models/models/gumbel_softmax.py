from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
import wandb

import torch
from torch.nn.modules.linear import Linear
from torch import nn
import torch.nn.functional as F

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)
from .transformer_binary_qa_model import TransformerBinaryQA
from .utils import safe_log, right_pad, batch_lookup, EPSILON

class GumbelSoftmaxModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tau = 1
        self.eps = 1e-10
        self.dim = -1

    def gs(self, logits, hard):
        return F.gumbel_softmax(
            logits, tau=self.tau, hard=hard, eps=self.eps, dim=self.dim
        )

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)

        # Compute straight through gumbel softmax
        y_soft = self.gs(output['pooled_output'], False)
        y_hard = self.gs(output['pooled_output'], True)
        output['gs'] = (y_hard - y_soft).detach() + y_soft
        
        return output
        

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
        self.qa_vocab = qa_model.vocab
        self.vocab = vocab
        self.dataset_reader = dataset_reader
        self.variant = variant
        self.sentence_embedding_method = sentence_embedding_method
        self.similarity_func = 'inner'       # TODO: set properly
        self.n_retrievals = 1                # TODO: set properly

        # Rollout params
        self.x = -111
        self.gamma = 1      # TODO
        self.beta = 1       # TODO
        self.n_mc = 5           # TODO
        self.num_rollout_steps = topk
        self.retriever_model = None
        self.run_analysis = False   # TODO
        self.baseline = 'n/a'
        self.training = True

        self.define_modules()
        
    def get_retrieval_distr(self, qr, c):
        # Compute embeddings
        e_q = self.get_query_embs(qr)['pooled_output']      # TODO: check against cls_output
        e_c = self.get_context_embs(c)

        # Compute similarities
        if self.similarity_func == 'inner':
            sim = torch.matmul(e_c, e_q.unsqueeze(-1)).squeeze()
        else:
            raise NotImplementedError()

        # Ensure padding receives 0 probability mass
        retrieval_mask = (c != self.retriever_pad_idx).long().unsqueeze(-1)
        similarity = torch.where(
            retrieval_mask.sum(dim=2).squeeze() == 0, 
            torch.tensor(-float("inf")).to(c.device), 
            sim,
        )

        # Policy is distribution over actions
        policy = F.softmax(similarity, dim=1)

        # Deal with nans- these are caused by all sentences being padding.
        # In this case, retrieve following uniform dist
        policy[torch.isnan(policy.sum(dim=1))] = 1 / policy.size(0)
        if torch.isnan(policy.sum(dim=1)).all():
            raise ValueError

        return similarity

    def answer(self, qr, label, metadata)->dict:
        return self.get_query_embs(qr, label, metadata)
        
    def get_query_embs(self, qr, label=None, metadata=None)->dict:
        qr_ = {'tokens': {'token_ids': qr, 'type_ids': torch.zeros_like(qr)}}
        return self.qa_model(qr_, label, metadata)

    def get_context_embs(self, c):
        return self.retriever_model(c)

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
    
    def forward(self, 
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        **kwargs,
    ) -> torch.Tensor:
        query = retrieval['tokens']['token_ids'][:,0,:]
        context = retrieval['tokens']['token_ids'][:,1:,:]
        self.device = query.device

        output = self.rollout(query, context, label, metadata)

    def stgs(self, logits):
        ''' Straight through gumbel softmax
        '''
        def gs(logits, hard):
            return F.gumbel_softmax(logits, tau=1, hard=hard, eps=1e-10, dim=-1)

        def fun(logits):
            l = [gs(logits, True)[0].nonzero().item() for _ in range(1000)]
            d = {x:l.count(x) for x in set(l)}
            # return sorted(list(d.keys()), key=lambda x: d[x], reverse=True)
            return d

        y_soft = gs(logits, False)
        y_hard = gs(logits, True)
        return (y_hard - y_soft).detach() + y_soft

    def rollout(self, qr, c, label, metadata):
        ''' Sample traces to obtain expected rewards for an episode.
            - qr: tokenized query + retrieval
            - c: tokenized context
            - label: gold labels
            - metadata: list of dicts
        '''
        # TODO: add an "end" context item of a blank one or something....

        # Storage tensors
        _d = self.device
        unnorm_policies = torch.full((self.num_rollout_steps, *c.shape[:-1]), self.x).to(_d)    # Shape: [num_steps, bsz, max_num_context, max_sentence_len]
        actions = torch.full_like(unnorm_policies, self.x)                
        # log_action_probs = torch.full_like(actions, self.x)                                     # Shape: [num_steps, bsz]
        
        # Retrieval rollout phase
        for t in range(self.num_rollout_steps - 1):
            unnorm_policies[t] = self.get_retrieval_distr(qr, c)
            actions[t] = self.stgs(unnorm_policies[t])
            new_idxs, c, metadata = self.prep_next_batch(c, metadata, actions, t)
            qr = torch.matmul(actions[t].unsqueeze(1), new_idxs)

        # Answer query
        self.update_meta(qr, metadata, actions)
        output = self.answer(qr, label, metadata)
        unnorm_policies[-1] = right_pad(output['label_probs'], unnorm_policies[-1])
        # actions[-1], log_action_probs[-1] = self.sample_action(unnorm_policies[-1])
        action = self.stgs(unnorm_policies[-1])


        # Record trajectory data
        output['unnorm_policies'] = unnorm_policies
        # output['log_action_probs'] = log_action_probs
        output['samples_actions'] = actions

        return output

    def update_meta(self, query_retrieval, metadata, actions):
        for qr, topk, meta in zip(query_retrieval, actions.T, metadata):
            meta['topk'] = topk.tolist()
            meta['query_retrieval'] = qr.tolist()

    def prep_next_batch(self, context, metadata, actions, t):
        ''' Retrieve the top k context sentences and prepare batch
            for use in the qa model.
        '''
        # Get indexes of retrieval items
        retrievals = actions.argmax(dim=-1).T

        # Concatenate query + retrival to make new query_retrieval
        # tensor of idxs        
        sentences = []
        for topk, meta in zip(retrievals, metadata):
            sentence_idxs = [int(i) for i in topk.tolist()[:t+1] if i != self.x]
            question = meta['question_text']
            context_str = ''.join([
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]).strip()
            meta['context_str'] = f"q: {question} c: {context_str}"
            sentences.append((question, context_str))
        batch = self.dataset_reader.transformer_indices_from_qa(sentences, self.qa_vocab)
        # src = batch['phrase']['tokens']['token_ids'].unsqueeze(1).to(_d)
        src = batch['phrase']['tokens']['token_ids'].unsqueeze(1)

        # Make tensor containing idxs of [query||retrieval]
        _d = context.device
        query_retrieval_ = torch.full((*context.shape[:2], src.size(-1)), self.x).to(_d)
        query_retrieval_.scatter_(
            1, retrievals[:,t].repeat(src.size(-1), 1, 1).T, src.float().to(_d)
        )             # [bsz, context items, max sentence len]

        # Replace retrieved context with padding so same context isn't retrieved twice
        current_action = actions[t].argmax(dim=1)
        context.scatter_(
            1, current_action.repeat(context.size(-1), 1, 1).T, self.retriever_pad_idx
        )

        return query_retrieval_, context, metadata

    def sample_action(self, policy):
        action = torch.multinomial(policy, 1)
        action_prob = batch_lookup(policy, action)
        return action.squeeze(), safe_log(action_prob)

    def predict(self):
        pass
