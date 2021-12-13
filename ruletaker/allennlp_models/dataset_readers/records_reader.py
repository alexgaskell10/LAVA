from typing import Dict, Any
import random, re, os, json, logging, pickle
from copy import deepcopy
import pickle as pkl

from torch import Tensor
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from .rule_reasoning_reader import RuleReasoningReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("records_reader")
class RecordsReader(RuleReasoningReader):
    """
    Parameters
    ----------
    """
    @overrides
    def _read(self, adv_path: str):
        return self._read_adv(adv_path)

    def _read_adv(self, file_path):

        with open(file_path, 'rb') as data_file:
            logger.info("Reading adversarial instances from pickle dataset at: %s", file_path)        
            records = pkl.load(data_file)

        n = 0
        max_instances = -1 if self.max_instances is None else self.max_instances // 2
        qids, qid_texts = {}, {}
        for record in records:
            if n == max_instances:
                break

            if not record['qa_fooled']:
                continue                # Only include incorrectly answered adversarial questions

            base_id = record['id']
            if base_id in qids:
                qids[base_id] += 1
            else:
                qids[base_id] = 1
            id = 'Adv-' + base_id + '-' + str(qids[base_id])

            context = '.'.join(record['sampled_sentences'][1:-1]) + '.'
            label = 1 - record['mod_label']     # The adversarial label (mod_label) = 1 - true label

            n += 1
            yield self.text_to_instance(
                    item_id=id,
                    question_text=record['question'],
                    context=context,
                    label=label,
                )
