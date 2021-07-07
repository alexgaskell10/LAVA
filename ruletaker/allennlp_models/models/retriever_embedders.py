import os
import sys

import torch
from torch import nn

from transformers import AutoModel

from allennlp.common.util import get_spacy_model


class BaseRetrievalEmbedder(nn.Module):
    def __init__(self, sentence_embedding_method, vocab, variant):
        super().__init__()
        self.sentence_embedding_method = sentence_embedding_method
        self.vocab = vocab
        self.retriever_pad_idx = self.vocab.get_token_index(self.vocab._padding_token)
        self.variant = variant
        self.init()

    def init(self):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class SpacyRetrievalEmbedder(BaseRetrievalEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
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
        self.embedder = nn.Embedding(*spacy_embs.shape)
        self.embedder.weight.data = spacy_embs

    def forward(self, idxs):
        ''' Compute sentence embeddings of input ids using chosen 
            embedding method.
        '''
        raise NotImplementedError
        # Compute token embeddings
        token_embs = self.embedder(idxs)    # [bsz, context_sentences, max_context_tokens, emb_dim]

        # Compute sentence embeddings
        retrieval_mask = (idxs != self.retriever_pad_idx).long().unsqueeze(-1)
        if self.sentence_embedding_method == 'mean':
            sentence_embs = (token_embs * retrieval_mask).sum(dim=2) / retrieval_mask.sum(dim=2)
        else:
            raise NotImplementedError()

        return sentence_embs


class TransformerRetrievalEmbedder(BaseRetrievalEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self):
        self.embedder = AutoModel.from_pretrained(self.variant)

    def forward(self, idxs):
        ''' Compute sentence embeddings of input ids using chosen 
            embedding method.
        '''
        # Prepare batch
        input_ids = idxs.contiguous().view(-1, idxs.size(-1))
        # mask = (input_ids != self.retriever_pad_idx).long()

        # Compute token embeddings
        # token_embs = self.embedder(input_ids, mask)[0]
        token_embs = self.embedder(input_ids)[0]
        
        # Use CLS token for sentence embedding
        sentence_embs = token_embs[:,0,:].squeeze().unsqueeze(0)

        return sentence_embs.view(*idxs.shape[:2], -1)