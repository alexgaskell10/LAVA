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

class PolicyNetwork(nn.Module):
    def __init__(self,
        qa_model: Model,
        variant: str,
        sentence_embedding_method: str = 'mean',
        vocab = None,
        dataset_reader = None,
    ):
        super().__init__()
        self.qa_model = qa_model
        self.qa_vocab = qa_model.vocab
        self.vocab = vocab
        self.dataset_reader = dataset_reader
        self.variant = variant
        self.sentence_embedding_method = sentence_embedding_method
        self.similarity_func = 'inner'       # TODO: set properly
        self.n_retrievals = 1         # TODO: set properly
        self.define_modules()
        
    def transit(self, qr, c):
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

        if torch.isinf(policy.sum(dim=1)).any():
            print('abc')

        # Deal with nans- these are caused by all sentences being padding.
        # In this case, retrieve following uniform dist
        policy[torch.isnan(policy.sum(dim=1))] = 1 / policy.size(0)       

        if torch.isnan(policy.sum(dim=1)).all():
            print('abc')

        return policy

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
    