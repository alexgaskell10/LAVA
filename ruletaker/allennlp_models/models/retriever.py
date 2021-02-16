from typing import Dict, Optional, List, Any
import logging
import os
import sys
import time
import wandb

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
from .transformer_binary_qa_model import TransformerBinaryQA

logger = logging.getLogger(__name__)

@Model.register("retrieval_scorer")
class RetrievalScorer(Model):
    """
    """
    def __init__(self,
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
        super().__init__(vocab, regularizer)
        self.vocab = vocab
        self.variant = variant
        self._predictions = []

        if variant == 'spacy':
            self.model = SpacyRetrievalEmbedder(
                sentence_embedding_method=sentence_embedding_method,
                vocab=self.vocab,
                variant=variant,
            )
            self.tok_name = 'tokens'
            self._output_dim = self.model.embedder.embedding_dim         # TODO: set this
        elif 'roberta' in variant:
            self.model = TransformerRetrievalEmbedder(
                sentence_embedding_method=sentence_embedding_method,
                vocab=self.vocab,
                variant=variant,
            )
            self.tok_name = 'token_ids'
            self._output_dim = self.model.embedder.config.hidden_size
        else:
            raise ValueError(
                f"Invalid retriever_variant = {variant}.\nInvestigate!"
            )

        # unifing all model classification layer
        self._classifier = Linear(self._output_dim, num_labels)
        self._classifier.weight.data.normal_(mean=0.0, std=0.02)
        self._classifier.bias.data.zero_()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self._debug = -1

    def forward(self, 
        phrase: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None,
        sentences: List = None,
    ) -> torch.Tensor:
        
        self._debug -= 1
        input_ids = phrase['tokens'][self.tok_name]

        model_output = self.model(input_ids.unsqueeze(1))
        label_logits = self._classifier(model_output.squeeze())

        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)

        if label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss      # TODO this is shortcut to get predictions fast..

            # Hack to use wandb logging
            if os.environ['WANDB_LOG'] == 'true':
                self.wandb_log(metadata, label_logits, label, loss)

            for e, example in enumerate(metadata):
                logits = sanitize(label_logits[e, :])
                label_probs = sanitize(output_dict['label_probs'][e, :])
                prediction = sanitize(output_dict['answer_index'][e])                    
                prediction_dict = {
                    'id': example['id'],
                    'phrase': example['question_text'],
                    'context': example['context'],
                    'logits': logits,
                    'label_probs': label_probs,
                    'answer': example['label'],
                    'prediction': prediction,
                    'is_correct': (example['label'] == prediction) * 1.0,
                    'q_depth': example['QDep'] if 'QDep' in example else None,
                    'retrievals': example['topk'] if 'topk' in example else None,
                }

                if 'skills' in example:
                    prediction_dict['skills'] = example['skills']
                if 'tags' in example:
                    prediction_dict['tags'] = example['tags']
                self._predictions.append(prediction_dict)

        return output_dict

    def wandb_log(self, metadata, label_logits, label, loss):
        prefix = 'train' if self.training else 'val'

        # Metrics by question depth
        if 'QDep' in metadata[0]:
            depth_accuracies = {}
            q_depths = torch.tensor([m['QDep'] for m in metadata]).to(label.device)
            for dep in q_depths.unique():
                idxs = (q_depths == dep).nonzero().squeeze()
                logits_ = label_logits[idxs]
                labels_ = label[idxs]
                ca = CategoricalAccuracy()
                ca(logits_, labels_)
                depth_accuracies[f"{prefix}_acc_{dep}"] = ca.get_metric()
            wandb.log(depth_accuracies, commit=False)

        # Aggregate metrics
        c = CategoricalAccuracy()
        c(label_logits, label)
        wandb.log({
            prefix+"_loss": loss, 
            prefix+"_acc": self._accuracy.get_metric(), 
            prefix+"_acc_noncuml": c.get_metric()
        })

    def decode(self, idxs):
        idx2tok = self.vocab._index_to_token if self.variant == 'spacy' else self.vocab._index_to_token['tags']
        return [idx2tok(i) for i in idxs.tolist()]

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset == True and not self.training:
            return {
                'EM': self._accuracy.get_metric(reset),
                'predictions': self._predictions,
            }
        else:
            return {
                'EM': self._accuracy.get_metric(reset),
            }