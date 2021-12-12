from typing import Dict, Any
import json
import logging
import random
import re
import pickle as pkl

from itertools import chain
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields import MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from .rule_reasoning_reader import RuleReasoningReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("blended_rule_reasoning")
class BlendedRuleReasoningReader(RuleReasoningReader):
    """
    Parameters
    ----------
    """

    def __init__(
        self,
        pretrained_model: str,
        max_pieces: int = 512,
        syntax: str = "rulebase",
        add_prefix: Dict[str, str] = None,
        skip_id_regex: str = None,
        scramble_context: bool = False,
        use_context_full: bool = False,
        sample: int = -1,
        max_instances = None,
        cache_directory = None,
        adversarial_examples_path_train = None,
        adversarial_examples_path_val = None,
        adversarial_examples_path_test = None,
    ) -> None:
        max_instances = None if max_instances == -1 else max_instances
        cache_directory = None if cache_directory == 'none' else cache_directory
        DatasetReader.__init__(self, max_instances=max_instances, cache_directory=cache_directory)

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces)
        self._tokenizer_internal = self._tokenizer.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._add_prefix = add_prefix
        self._scramble_context = scramble_context
        self._use_context_full = use_context_full
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex
        self._adv_train_path = None if adversarial_examples_path_train == 'none' else adversarial_examples_path_train
        self._adv_val_path = None if adversarial_examples_path_val == 'none' else adversarial_examples_path_val
        self._adv_test_path = None if adversarial_examples_path_test == 'none' else adversarial_examples_path_test

        tok = type(self).__name__
        adv_file = self._adv_train_path.split('/')[3] if self._adv_train_path is not None else self._adv_train_path
        self.pkl_file = f'{tok}_{self._max_pieces}_{adv_file}_{max_instances}_DSET.pkl'

        self._original_only = False
        self._adversarial_only = False

    @overrides
    def _read(self, file_path: str):
        if self._adversarial_only:
            instances = iter(())
        else:
            instances = self._read_internal(file_path)

        if not self._original_only and 'train' in file_path and self._adv_train_path is not None:
            adv_instances = self._read_adv(self._adv_train_path)
        elif not self._original_only and 'dev' in file_path and self._adv_val_path is not None:
            adv_instances = self._read_adv(self._adv_val_path)
        elif not self._original_only and 'test' in file_path and self._adv_test_path is not None:
            adv_instances = self._read_adv(self._adv_test_path)
        else:
            adv_instances = iter(())

        return chain(instances, adv_instances)

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

            # if base_id in qid_texts and qid_texts[base_id] == record['question']:
            #     continue            # Skip 
            # else:
            #     qid_texts[base_id] = record['question']

            context = '.'.join(record['sampled_sentences'][1:-1]) + '.'
            label = 1 - record['mod_label']     # The adversarial label (mod_label) = 1 - true label

            n += 1
            yield self.text_to_instance(
                    item_id=id,
                    question_text=record['question'],
                    context=context,
                    label=label,
                )

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5
        is_done = False

        n = 0
        max_instances = -1 if self.max_instances is None else self.max_instances // 2
        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                if n == max_instances:
                    break
                if is_done:
                    break
                item_json = json.loads(line.strip())
                item_id = item_json.get("id", "NA")
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                if self._syntax == "rulebase":
                    questions = item_json['questions']
                    if self._use_context_full:
                        context = item_json.get('context_full', '')
                    else:
                        context = item_json.get('context', "")
                elif self._syntax == "propositional-meta":
                    questions = item_json['questions'].items()
                    sentences = [x['text'] for x in item_json['triples'].values()] + \
                                [x['text'] for x in item_json['rules'].values()]
                    if self._scramble_context:
                        random.shuffle(sentences)
                    context = " ".join(sentences)
                else:
                    raise ValueError(f"Unknown syntax {self._syntax}")

                for question in questions:
                    counter -= 1
                    debug -= 1
                    if n == max_instances:
                        break
                    if counter == 0:
                        is_done = True
                        break
                    if debug > 0:
                        logger.info(item_json)
                    if self._syntax == "rulebase":
                        text = question['text']
                        q_id = question.get('id')
                        label = None
                        if 'label' in question:
                            label = 1 if question['label'] else 0
                    elif self._syntax == "propositional-meta":
                        text = question[1]['question']
                        q_id = f"{item_id}-{question[0]}"
                        label = question[1].get('propAnswer')
                        if label is not None:
                            label = ["False", "True", "Unknown"].index(label)
                    n += 1
                    yield self.text_to_instance(
                        item_id=q_id,
                        question_text=text,
                        context=context,
                        label=label,
                        debug=debug)

    def adv_path(self, dataset):
        if 'val' in dataset:
            return self._adv_val_path
        elif 'test' in dataset:
            return self._adv_test_path
        elif 'train' in dataset:
            return self._adv_train_path

    def set_original_only(self, bool):
        self._original_only = bool

    def set_adversarial_only(self, bool):
        self._adversarial_only = bool
