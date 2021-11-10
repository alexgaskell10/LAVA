from typing import Dict, Optional, List, Any
import logging

from transformers.modeling_t5 import T5Model
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_xlnet import XLNetModel
from transformers.modeling_bert import BertModel
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import SequenceSummary
import re
import json
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy

import wandb
import os

logger = logging.getLogger(__name__)

@Model.register("transformer_binary_qa")
class TransformerBinaryQA(Model):
    """
    """
    def __init__(
        self,
        vocab: Vocabulary,
        pretrained_model: str = None,
        requires_grad: bool = True,
        transformer_weights_model: str = None,
        num_labels: int = 2,
        predictions_file=None,
        layer_freeze_regexes: List[str] = None,
        regularizer: Optional[RegularizerApplicator] = None
    ):
        super().__init__(vocab, regularizer)

        self._predictions = []

        self._pretrained_model = pretrained_model

        if 't5' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            if transformer_weights_model:  # Override for RoBERTa only for now
                logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
                transformer_model_loaded = load_archive(transformer_weights_model)
                self._transformer_model = transformer_model_loaded.model._transformer_model
            else:
                self._transformer_model = T5Model.from_pretrained(pretrained_model)
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        if 'roberta' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            if transformer_weights_model:  # Override for RoBERTa only for now
                logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
                transformer_model_loaded = load_archive(transformer_weights_model)
                self._transformer_model = transformer_model_loaded.model._transformer_model
            else:
                self._transformer_model = RobertaModel.from_pretrained(pretrained_model)
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'xlnet' in pretrained_model:
            self._padding_value = 5  # The index of the XLNet padding token
            self._transformer_model = XLNetModel.from_pretrained(pretrained_model)
            self.sequence_summary = SequenceSummary(self._transformer_model.config)
        elif 'albert' in pretrained_model:
            self._transformer_model = AlbertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'bert' in pretrained_model:
            self._transformer_model = BertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        else:
            assert ValueError

        for name, param in self._transformer_model.named_parameters():
            if layer_freeze_regexes and requires_grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            else:
                grad = requires_grad
            if grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        transformer_config = self._transformer_model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self._transformer_model.config.hidden_size

        # unifing all model classification layer
        self._classifier = Linear(self._output_dim, num_labels)
        self._classifier.weight.data.normal_(mean=0.0, std=0.02)
        self._classifier.bias.data.zero_()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self._debug = -1

    def forward(self, 
            phrase,#: Dict[str, torch.LongTensor],
            label: torch.LongTensor = None,
            metadata: List[Dict[str, Any]] = None,
            index_tensor: torch.Tensor = None,
        ) -> torch.Tensor:

        self._debug -= 1
        input_ids = phrase['tokens']['token_ids']       # TODO sort for both forward passes
        segment_ids = phrase['tokens']['type_ids']

        question_mask = (input_ids != self._padding_value).long()

        # Segment ids are not used by RoBERTa
        if 'roberta' in self._pretrained_model or 't5' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(
                input_ids=util.combine_initial_dims(input_ids),
                # token_type_ids=util.combine_initial_dims(segment_ids),
                attention_mask=util.combine_initial_dims(question_mask),
            )
            cls_output = self._dropout(pooled_output)
        if 'albert' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(
                input_ids=util.combine_initial_dims(input_ids),
                # token_type_ids=util.combine_initial_dims(segment_ids),
                attention_mask=util.combine_initial_dims(question_mask)
            )
            cls_output = self._dropout(pooled_output)
        elif 'xlnet' in self._pretrained_model:
            transformer_outputs = self._transformer_model(
                input_ids=util.combine_initial_dims(input_ids),
                token_type_ids=util.combine_initial_dims(segment_ids),
                attention_mask=util.combine_initial_dims(question_mask)
            )
            cls_output = self.sequence_summary(transformer_outputs[0])
        elif 'bert' in self._pretrained_model:
            last_layer, pooled_output = self._transformer_model(
                input_ids=util.combine_initial_dims(input_ids),
                token_type_ids=util.combine_initial_dims(segment_ids),
                attention_mask=util.combine_initial_dims(question_mask)
            )
            cls_output = self._dropout(pooled_output)
        else:
            assert (ValueError)

        label_logits = self._classifier(cls_output)

        if label_logits.size(1) == 2:
            label_logits_ = label_logits
        elif label_logits.size(1) == 1:
            label_logits_ = torch.cat([(1-label_logits.sigmoid()).log(), label_logits.sigmoid().log()], axis=1)
            assert ((label_logits_.softmax(-1)[:,1:2] - label_logits.sigmoid()) < 1e-5).all()

        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits_, dim=1)
        output_dict['answer_index'] = label_logits_.argmax(1)
        output_dict['cls_output'] = cls_output
        output_dict['pooled_output'] = pooled_output

        if label is not None:
            loss = self._loss(label_logits_, label)
            self._accuracy(label_logits_, label)
            output_dict["loss"] = loss
            output_dict["label"] = label

            # Hack to use wandb logging
            if getattr(os.environ, "WANDB_LOG", "false") == 'true':
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
                    'q_length': example['QDep'] if 'QLen' in example else None,
                    'retrievals': example['topk'] if 'topk' in example else None,
                    'retrieval_recall': self.retrieval_recall(example) if 'node_label' in example and 'topk' in example else None
                }

                if 'skills' in example:
                    prediction_dict['skills'] = example['skills']
                if 'tags' in example:
                    prediction_dict['tags'] = example['tags']
                self._predictions.append(prediction_dict)

        return output_dict

    def wandb_log(self, metadata, label_logits, label, loss):
        prefix = 'train' if self.training else 'val'

        # Metrics by proof length
        if 'QLen' in metadata[0]:
            len_accs = {}
            retrieval_recalls = {}
            q_lens = torch.tensor([m['QLen'] for m in metadata]).to(label.device)
            for len_ in q_lens.unique():
                idxs = (q_lens == len_).nonzero().squeeze()

                # Accuracy
                logits_ = label_logits[idxs]
                labels_ = label[idxs]
                ca = CategoricalAccuracy()
                ca(logits_, labels_)
                len_accs[f"{prefix}_acc_{len_}"] = ca.get_metric()

                # Retrieval recall
                meta = [metadata[i] for i in (idxs if idxs.dim() else idxs.unsqueeze(0)).tolist()]
                retrieval_recalls[f"{prefix}_ret_recall_{len_}"] = self.batch_retrieval_recall(meta)

            wandb.log({**len_accs, **retrieval_recalls}, commit=False)

        # Aggregate metrics
        c = CategoricalAccuracy()
        c(label_logits, label)
        wandb.log({
            prefix+"_loss": loss.mean(), 
            prefix+"_acc": self._accuracy.get_metric(), 
            prefix+"_acc_noncuml": c.get_metric(),
            prefix+"_ret_recall": self.batch_retrieval_recall(metadata),
        })

    def retrieval_recall(self, example):
        proof_idxs = {n for n,i in enumerate(example['node_label'][:-1]) if i == 1}
        if example['label'] and proof_idxs:
            correct_retrieval_idxs = proof_idxs & set(example['topk'])
            return len(correct_retrieval_idxs) / len(proof_idxs)
        else:
            return -1

    def batch_retrieval_recall(self, metadata):
        recalls = []
        for example in metadata:
            if example['label']:
                recall = self.retrieval_recall(example)
                if recall != -1:
                    recalls.append(recall)
        return sum(recalls) / len(recalls) if recalls else None

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
