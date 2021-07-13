from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time

from transformers.tokenization_auto import AutoTokenizer
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

from transformers import AutoModel

from .retriever_embedders import (
    SpacyRetrievalEmbedder, TransformerRetrievalEmbedder
)
from .utils import safe_log, right_pad, batch_lookup, EPSILON, make_dot, set_dropout, one_hot, lmap, lfilter
from .transformer_binary_qa_model import TransformerBinaryQA
from .baseline import Baseline

torch.manual_seed(0)

@Model.register("variational_inference_base")
class ELBO(Model):
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
        self.variant = variant
        self.qa_model = qa_model        # TODO: replace with fresh transformerbinaryqa
        self.qa_model._loss = nn.CrossEntropyLoss(reduction='none')
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self.qa_vocab = qa_model.vocab
        self.dataset_reader = dataset_reader
        self.infr_model = InferenceNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader)
        self.gen_model = GenerativeNetwork(variant=variant, vocab=vocab, dataset_reader=dataset_reader)
        self.vocab = vocab
        self.regularizer = regularizer
        self.sentence_embedding_method = sentence_embedding_method
        self.n_z = topk
        # self.kl_div = nn.KLDivLoss(reduction='none')
        self._beta = 1

    def forward(self,
        phrase=None, label=None, metadata=None, retrieval=None, **kwargs,
    ) -> torch.Tensor:
        ''' Forward pass of the network. 

            - p(e|z,q) - answering term (qa_model)
            - q(z|e,q,c) - variational distribution (infr_model)
            - p(z|q,c) - generative distribution (gen_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
        '''
        self._d = phrase['tokens']['token_ids'].device

        infr_logits = self.infr_model(phrase, label)
        gen_logits = self.gen_model(phrase)
        # TODO: make multi-label classification problem (so sigmoid rather than softmax output layer)
        z = self._draw_samples(infr_logits)
        batch = self._prep_batch(z, metadata, label)
        qa_output = self.qa_model(**batch)

        # Compute log probabilites from logits and sample
        infr_logprobs = -self._loss(infr_logits, z.squeeze(-1))
        gen_logprobs = -self._loss(gen_logits, z.squeeze(-1))

        # kl_div = self.kl_div(infr_logits.log_softmax(-1), gen_logits.softmax(-1)).sum(-1)
        qa_logprobs = -qa_output['loss'].detach()
        elbo = qa_logprobs - self._beta * (infr_logprobs - gen_logprobs)
        outputs = {"loss": -elbo.mean()}
        return outputs

    def _prep_batch(self, z, metadata, label):
        ''' Concatenate the latest retrieval to the current 
            query+retrievals. Also update the tensors for the next
            rollout pass.
        '''
        # Concatenate query + retrival to make new query_retrieval matrix of idxs        
        sentences = []
        for topk, meta, e in zip(z, metadata, label):
            question = meta['question_text']
            sentence_idxs = topk.tolist()
            context_rtr = [
                toks + '.' for n, toks in enumerate(meta['context'].split('.')[:-1]) 
                if n in sentence_idxs
            ]
            meta['context_str'] = f"q: {question} c: {''.join(context_rtr).strip()}"
            sentences.append((question, ''.join(context_rtr).strip(), e))

        batch = self.dataset_reader.encode_batch(sentences, self.qa_vocab)
        return self.dataset_reader.move(batch, self._d)

    def _gs(self, logits, tau=1):
        ''' Sample using Gumbel Softmax. Ingests raw logits.
        '''
        return F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)

    def _draw_samples(self, p):
        ''' Obtain samples from a distribution
            - p: probability distribution
        '''
        samples = torch.zeros(p.size(0), self.n_z, dtype=torch.long).to(self._d)
        # TODO: look into this - don't think this is a valid thing to do...
        for i in range(self.n_z):
            samples[:, i] = self._gs(p).argmax(-1)
            # TODO: mask out already sampled values
        return samples


class _BaseSentenceClassifier(Model):
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
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
        self._W = Linear(self._output_dim * 2, num_labels)
        self._W.weight.data.normal_(mean=0.0, std=0.02)
        self._W.bias.data.zero_()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        # Split sentences in the context based on full stop
        self.split_idx = self.dataset_reader.encode_token('.', mode='retriever')

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

        # TODO: experiment with concatenating first and last token embs from sentence + mean pooling
        # Concat first and last token idxs
        max_num_sentences = (x == self.split_idx).nonzero()[:,0].bincount().max() - 1
        node_reprs = torch.full((x.size(0), max_num_sentences), 0.).to(self._d)      # shape: (bsz, # sentences, 2, model_dim)
        for b in range(x.size(0)):
            # Create list of end idxs of each context item
            end_idxs = (x[b] == self.split_idx).nonzero().squeeze().tolist()
            q_end = end_idxs.pop(0)     # Remove first list item as it is the question
            end_idxs.insert(0, q_end + 5)   # Tokenizer adds five "decorative" tokens at the beginning of the context

            # Form tensor containing embedding of first and last token for each sentence
            reprs = torch.zeros(len(end_idxs)-1, 2, embs.size(-1)).to(self._d)      # shape: (# context items, 2, model_dim)
            for i in range(1, len(end_idxs)):
                start_idx = end_idxs[i-1]
                end_idx = end_idxs[i] - 1           # -1 as this ignores the full stops
                
                reprs[i-1, 0] = embs[b, start_idx]
                reprs[i-1, 1] = embs[b, end_idx]

            # Pass through classifier
            reprs_ = reprs.view(reprs.size(0), -1)
            node_logits = self._W(reprs_).squeeze(-1)                
            node_reprs[b, :len(node_logits)] = node_logits

        return node_reprs.log_softmax(-1)


class InferenceNetwork(_BaseSentenceClassifier):
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
        super().__init__(variant, vocab, dataset_reader, regularizer, num_labels)
        self.e_true = torch.tensor(
            [self.dataset_reader.encode_token(tok, mode='retriever') for tok in '<s> ĠE : ĠTrue </s>'.split()]
        )
        self.e_false = torch.tensor(
            [self.dataset_reader.encode_token(tok, mode='retriever') for tok in '<s> ĠE : ĠFalse </s>'.split()]
        )

    def forward(self, phrase, label, **kwargs) -> torch.Tensor:
        ''' Forward pass of the network. Outputs a distribution over sentences z. 

            - q(z|e,q,c) - variational distribution (infr_model)
            - e: entailment relation [0,1] (i.e. does q follow from c (or z))
            - z: retrievals (subset of c)
            - c: context (rules + facts)
            - q: claim
        '''
        e = label
        qc = phrase['tokens']['token_ids']       # shape = (bsz, context_len)
        self._d = qc.device

        # Prepare model inputs by encoding and concatenating e
        eqc = self._encode_and_append_label(qc, e)

        return super().forward(eqc)

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
                eqc[b, :len_e] = self.e_true.to(self._d)
            elif label[b] == 0:
                eqc[b, :len_e] = self.e_false.to(self._d)
            else:
                raise ValueError
                
        return eqc


class GenerativeNetwork(_BaseSentenceClassifier):
    def __init__(self, variant, vocab, dataset_reader, regularizer=None, num_labels=1):
        super().__init__(variant, vocab, dataset_reader, regularizer, num_labels)

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
