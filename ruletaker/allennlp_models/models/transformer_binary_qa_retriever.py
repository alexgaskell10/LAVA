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
from allennlp.common.util import get_spacy_model

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
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:
        super().__init__(qa_model.vocab, regularizer)

        if variant == 'spacy':
            self.retriever = get_spacy_model(
                spacy_model_name="en_core_web_md", pos_tags=False, parse=False, ner=False
            )
        else:
            raise ValueError(
                f"Invalid retriever_variant = {retriever_variant}.\nInvestigate!"
            )

        self.vocab = vocab
        self.qa_model = qa_model

        self._debug = -1

    def forward(self, 
        phrase: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        retrieval: List = None, 
    ) -> torch.Tensor:

        retrieval_idxs = retrieval['tokens']['tokens']
        device = retrieval_idxs.device
        retrieval_mask = retrieval['tokens']['mask'] if 'mask' in retrieval['tokens'] else torch.where(
            retrieval_idxs > 0,                                             # TODO: replace 0 with padding index
            torch.tensor(1).to(device), torch.tensor(0).to(device)
        )

        with torch.no_grad():
            # TODO: add aggr type (mean, sum etc)
            
            
            # Broken here
            token_embs = self.retriever(retrieval_idxs).tokens
            sentence_embs = (y*mask).sum(dim=0)/mask.sum(dim=0)

        # return output_dict


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

