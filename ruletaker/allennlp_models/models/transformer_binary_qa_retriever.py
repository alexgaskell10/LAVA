from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time

import torch
from torch.nn.modules.linear import Linear
from torch import nn

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)

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
        dataset_reader = None,
        pretrained_retriever_model = None,
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)
        self.qa_vocab = qa_model.vocab
        self.vocab = vocab
        self.qa_model = qa_model
        self.topk = topk
        self.variant = variant
        self.dataset_reader = dataset_reader
        self.retriever_model = None

        # Load pretrained retriever
        if pretrained_retriever_model is not None:
            retriever_archive = load_archive(
                archive_file=pretrained_retriever_model,
                cuda_device=0   #qa_model.device,   # TODO: sort this
            )
            self.retriever_model = retriever_archive.model
            self.similarity = None

        if variant == 'spacy':
            if self.retriever_model is None:
                self.retriever_model = SpacyRetrievalEmbedder(
                    sentence_embedding_method=sentence_embedding_method,
                    vocab=self.vocab,
                    variant=variant,
                )
                self.similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
                
            self.tok_name = 'tokens'
            self.retriever_pad_idx = self.vocab.get_token_index(self.vocab._padding_token)      # TODO: standardize these
        elif 'roberta' in variant:
            if self.retriever_model is None:
                self.retriever_model = TransformerRetrievalEmbedder(
                    sentence_embedding_method=sentence_embedding_method,
                    vocab=self.vocab,
                    variant=variant,
                )
                self.similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

            self.tok_name = 'token_ids'
            self.retriever_pad_idx = self.dataset_reader.pad_idx(mode='retriever')       # TODO: standardize these
        else:
            raise ValueError(
                f"Invalid retriever_variant = {variant}.\nInvestigate!"
            )

        self._debug = -1

    def forward(self, 
        phrase: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        sentences: List = None,
    ) -> torch.Tensor:

        phrase_idxs = sentences['tokens']['token_ids']
        retrieval_idxs = retrieval['tokens'][self.tok_name]
        self.device = phrase_idxs.device

        # Retrieve k ids of the most similar context sentences 
        # to the question
        topk_idxs = self.retrieve_topk_idxs(retrieval_idxs, metadata)

        # Use the top k indices to prepare a batch
        batch = self.prepare_retrieved_batch(topk_idxs, metadata)

        # Update metadata to include retrieval info
        for n, meta in enumerate(metadata):
            meta.update(
                topk=topk_idxs[n].tolist(), 
                retrieved_context=batch['metadata'][n]['context']
            )

        return self.qa_model.forward(
            phrase=batch['phrase'],
            label=label,
            metadata=metadata,
        )

    def retrieve_topk_idxs(self, idxs, metadata=None):
        ''' Use the specified retrieval embedder and retrieval method
            to find the top k most similar context sentences to the
            query.

        '''
        # Perform retrieval with no gradient
        with torch.no_grad():
            if self.similarity is not None:
                # Here the retriever is not trained on the retrieval task.
                # Similarity is computed using a similarity measure
                # (e.g. cosine similarity) between the embedded representations
                # of the context and question.

                # Compute sentence embeddings
                sentence_embs = self.retriever_model(idxs)

                # Compute similarity between context and query sentence embeddingss
                query, context = sentence_embs[:,:1,:], sentence_embs[:,1:,:]
                similarity = self.similarity(query, context)

                # Replace nans with 0
                retrieval_mask = (idxs != self.retriever_pad_idx).long().unsqueeze(-1)
                similarity = torch.where(
                    retrieval_mask[:,1:,:].sum(dim=2).squeeze() == 0,
                    torch.tensor(0).float().to(self.device), 
                    similarity,
                )
            else:
                # Here the retriever is trained on the retrieval task.
                # Similarity is computed by concatenatating the question 
                # with each context item and projecting the embedding 
                # represenation to a single dimension.
            
                # Compute similarity
                idxs_ = idxs.view(-1, idxs.size(-1))
                logits = self.retriever_model(idxs_)['label_probs']
                similarity = logits[:,1].view(idxs.shape[:2])

                # Replace nans with 0
                retrieval_mask = (idxs != self.retriever_pad_idx).long().unsqueeze(-1)
                similarity = torch.where(
                    retrieval_mask.sum(dim=2).squeeze() == 0,
                    torch.tensor(0).float().to(self.device), 
                    similarity,
                )

            # Find indices of k most similar context sentences
            topk = min(self.topk, similarity.size(1))
            topk_idxs = torch.topk(similarity, topk).indices
        
        return topk_idxs

    def prepare_retrieved_batch(self, topk_idxs, metadata):
        ''' Retrieve the top k context sentences and prepare batch
            for use in the qa model.
        '''
        sentences = []
        for topk, meta in zip(topk_idxs, metadata):
            sentence_idxs = topk.tolist()
            question = meta['question_text']
            context = ''.join([
                toks + '.' for n,toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]).strip()
            sentences.append((question, context))

        batch = self.dataset_reader.transformer_indices_from_qa(sentences, self.qa_vocab)
        batch['phrase']['tokens'] = {k:v.to(self.device) for k,v in batch['phrase']['tokens'].items()}
        return batch

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
