from typing import Dict, Optional, List, Any
import logging
import re
import json

from transformers.modeling_t5 import T5Model
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_xlnet import XLNetModel
from transformers.modeling_bert import BertModel
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import SequenceSummary

import torch
from torch.nn.modules.linear import Linear
from torch import nn

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

import wandb
import os
from allennlp.common.util import get_spacy_model
import time

logger = logging.getLogger(__name__)

@Model.register("transformer_binary_qa_retriever")
class TransformerBinaryQARetriever(Model):
    """
    """
    def __init__(self,
        qa_model: Model,
        variant: str,
        vocab: Vocabulary = None,
        # pretrained_model: str = None,
        requires_grad: bool = True,
        # transformer_weights_model: str = None,
        num_labels: int = 2,
        predictions_file=None,
        layer_freeze_regexes: List[str] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        topk: int = 5,
        sentence_embedding_method: str = 'mean',
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.vocab = vocab
        self.qa_model = qa_model
        self.topk = topk
        self.sentence_embedding_method = sentence_embedding_method

        if variant == 'spacy':
            self.retriever_embs = self.init_spacy()
            self.similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        else:
            raise ValueError(
                f"Invalid retriever_variant = {retriever_variant}.\nInvestigate!"
            )

        self.do_setup = True
        self._debug = -1

    def forward(self, 
        phrase: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        sentences: List = None,
    ) -> torch.Tensor:

        phrase_idxs = sentences['tokens']['token_ids']
        retrieval_idxs = retrieval['tokens']['tokens']
        self.device = device = phrase_idxs.device

        if self.do_setup:
            self.setup()

        topk_idxs = self.retrieve_topk_idxs(retrieval_idxs)

        # Construct context from topk indices
        # NOTE: appears that torch does not allow multidimensional indexing
        # using tensor so flatten to 2d before indexing
        query_idxs, context_idxs = phrase_idxs[:,:1,:], phrase_idxs[:,1:,:]
        topk_idxs_ = (
            topk_idxs + torch.arange(topk_idxs.size(0)).unsqueeze(1).to(device) * context_idxs.size(1)
        ).flatten()
        context_idxs_ = context_idxs.contiguous().view(-1, context_idxs.size(-1))
        topk_context_idxs = context_idxs_[topk_idxs_].view(
            context_idxs.size(0), topk_idxs.size(1), context_idxs.size(-1)
        )

        # Add special tokens and join to single tensor
        query_idxs_, topk_context_idxs_ = self.add_special_tokens(query_idxs, topk_context_idxs)
        theory_idxs = torch.cat((query_idxs_, topk_context_idxs_), dim=1)

        # Need to move padding from the middle of the context 
        # matrix to the end for each row
        index_ = torch.where(
            # If the entry is padding...
            theory_idxs.view(theory_idxs.size(0), -1) == self.qa_pad_idx,
            # ... replace with large number so can be sorted to the end for each dimension ...
            torch.tensor(theory_idxs.numel() + 1).to(device),
            # ... otherwise give index of entry (to retain current ordering)
            torch.arange(theory_idxs.numel()).view(theory_idxs.size(0), -1).to(device),
        )
        padding2end = index_.sort(dim=1).indices
        padding2end += torch.arange(index_.size(0)).unsqueeze(1).to(device) * index_.size(1)      # Hack as indices reset across dimensions...
        sorted_theory_idxs = theory_idxs.flatten()[padding2end.flatten()].view(
            theory_idxs.size(0), -1
        )

        # Check padding
        max_seq_len = self.check_correctly_padded(sorted_theory_idxs)
        # Truncate to max sequence length
        input_ids = sorted_theory_idxs[:, :max_seq_len + 1]     # TODO: check this

        # Reconfigure args for qa model forward pass
        phrase['tokens'] = {
            'token_ids': input_ids,
            'mask': input_ids != self.qa_pad_idx,
            'type_ids': torch.zeros_like(input_ids),     # TODO: automate for other model types
        }
        for n, meta in enumerate(metadata):
            meta.update(topk=topk_idxs[n].tolist())

        return self.qa_model.forward(
            phrase = phrase,
            label = label,
            metadata = metadata,
        )

    def check_correctly_padded(self, idxs):
        ''' Check that the padding has been added correctly
            (after tokens).
        '''
        # Check that padding index is at the end
        pad_idxs = (idxs == self.qa_pad_idx).nonzero()
        max_seq_len = 0
        for dim in pad_idxs[:,0].unique():
            idxs = pad_idxs[(pad_idxs[:,0] == dim).nonzero().flatten(), 1]
            assert torch.all(idxs == torch.arange(idxs.min(), idxs.max()+1).to(self.device))
            max_seq_len = idxs.min() if idxs.min() > max_seq_len else max_seq_len

        return max_seq_len

    def init_spacy(self):
        ''' Load spacy embeddings as nn.Embedding.
        '''
        spacy = get_spacy_model(
            spacy_model_name="en_core_web_md", pos_tags=False, parse=False, ner=False
        )
        idx2tok = self.vocab.get_index_to_token_vocabulary()
        spacy_vecs = {
            idx: torch.tensor(spacy(tok).vector).unsqueeze(0) for idx, tok in idx2tok.items()
        }
        spacy_embs = torch.cat(list(spacy_vecs.values()))
        retriever_embs = nn.Embedding(*spacy_embs.shape)
        retriever_embs.weight.data = spacy_embs
        return retriever_embs

    def retrieve_topk_idxs(self, idxs):
        ''' Use the specified retrieval embedder and retrieval method
            to find the top k most similar context sentences to the
            query.
        '''
        # Perform retrieval with no gradient
        with torch.no_grad():
            token_embs = self.retriever_embs(idxs)    # [bsz, context_sentences, max_context_tokens, emb_dim]

            # Compute sentence embeddings
            retrieval_mask = (idxs != self.retriever_pad_idx).long().unsqueeze(-1)
            if self.sentence_embedding_method == 'mean':
                sentence_embs = (token_embs * retrieval_mask).sum(dim=2) / retrieval_mask.sum(dim=2)
            else:
                raise NotImplementedError()

            # Compute similarity between context and query sentence embeddingss
            query, context = sentence_embs[:,:1,:], sentence_embs[:,1:,:]
            similarity = self.similarity(query, context)

            # Replace nans with 0
            similarity = torch.where(
                retrieval_mask[:,1:,:].sum(dim=2).squeeze() == 0,
                torch.tensor(0).float().to(self.device), 
                similarity,
            )

            # Find indices of k most similar context sentences
            topk = min(self.topk, similarity.size(1))
            topk_idxs = torch.topk(similarity, topk).indices
        
        return topk_idxs

    def setup(self):
        ''' Set attributes for forward pass. Should be set during
            first forward pass. 
        '''
        self.retriever_pad_idx = self.vocab.get_token_index(self.vocab._padding_token)
        self.qa_tok2idx = self.vocab._token_to_index['tags']
        self.qa_idx2tok = self.vocab._index_to_token['tags']
        # TODO set below dynamically
        self.qa_pad_idx = self.qa_tok2idx['<pad>']
        self.start_idx = self.qa_tok2idx['<s>']      
        self.end_idx = self.qa_tok2idx['</s>']
        self.q_idx = self.qa_tok2idx['ĠQ']
        self.c_idx = self.qa_tok2idx['ĠC']
        self.colon_idx = self.qa_tok2idx[':']
        self.start_query_idxs = torch.tensor(
            [self.start_idx, self.q_idx, self.colon_idx]
        ).to(self.device)
        self.start_context_idxs = torch.tensor(
            [self.start_idx, self.c_idx, self.colon_idx]
        ).to(self.device)
        self.end_idxs = torch.tensor(self.end_idx).to(self.device)
        self.num_extra = self.start_query_idxs.numel() + self.end_idxs.numel()
        self.do_setup = False

    def add_special_tokens(self, query_idxs, context_idxs):
        ''' Add the special tokens which are normally added at tokenization
            (e.g. <s>, </s>, 'ĠQ', 'ĠC', ':)
            TODO: automate this so works for other models
        '''
        # First remove special tokens (as only partially complete)
        query_idxs[query_idxs == self.start_idx] = self.qa_pad_idx
        query_idxs[query_idxs == self.end_idx] = self.qa_pad_idx
        context_idxs[context_idxs == self.start_idx] = self.qa_pad_idx
        context_idxs[context_idxs == self.end_idx] = self.qa_pad_idx

        # For query first
        query_idxs_ = torch.full((
            query_idxs.size(0), query_idxs.size(1), query_idxs.size(2) + self.num_extra
        ), self.qa_pad_idx).to(self.device)

        start_toks = self.start_query_idxs.numel()
        end_toks = self.end_idxs.numel()

        # Add start and end tokens
        query_idxs_[:,:,:start_toks] = self.start_query_idxs
        query_idxs_[:,:,-end_toks:] = self.end_idxs
        # Fill with query tokens
        query_idxs_[:,:,start_toks:-end_toks] = query_idxs

        # Now for context. Note that we add beginning and end of sequence tokens
        # around all context
        context_idxs_ = torch.full((
            context_idxs.size(0), context_idxs.size(1), query_idxs_.size(2)
        ), self.qa_pad_idx).to(self.device)

        start_toks = self.start_context_idxs.numel()

        # Add start and end tokens
        context_idxs_[:,0,:start_toks] = self.start_context_idxs
        context_idxs_[:,-1:,-end_toks:] = self.end_idxs
        # Fill with context tokens
        context_idxs_[:,:,start_toks:-end_toks] = context_idxs[:,:,:]

        return query_idxs_.long(), context_idxs_.long()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset == True and not self.training:
            return {
                'EM': self.qa_model._accuracy.get_metric(reset),
                'predictions': self.qa_model._predictions,
            }
        else:
            return {
                'EM': self.qa_model._accuracy.get_metric(reset),
            }

