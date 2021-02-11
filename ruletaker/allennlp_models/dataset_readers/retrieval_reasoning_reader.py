from typing import Dict, Any
import json
import logging
import random
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field, TextField, LabelField, MetadataField, SequenceLabelField,
    ListField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer, SpacyTokenizer
from allennlp.data.dataloader import allennlp_collate

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("retriever_reasoning")
class RetrievalReasoningReader(DatasetReader):
    """
    Parameters
    ----------
    """

    def __init__(self,
        pretrained_model: str = None,
        max_pieces: int = 512,
        syntax: str = "rulebase",
        add_prefix: Dict[str, str] = None,
        skip_id_regex: str = None,
        scramble_context: bool = False,
        use_context_full: bool = False,
        sample: int = -1,
        retriever_variant: str = None,
    ) -> None:
        super().__init__()
        
        # Init reasoning tokenizer
        self._tokenizer_qamodel = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces)
        self._tokenizer_qamodel_internal = self._tokenizer_qamodel.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers_qamodel = {'tokens': token_indexer}
        
        # Initalize retriever tokenizer
        if retriever_variant == 'spacy':
            self._tokenizer_retriever = SpacyTokenizer()
            self._token_indexers_retriever = {"tokens": SingleIdTokenIndexer()}
        elif 'roberta' in retriever_variant:
            self._tokenizer_retriever = PretrainedTransformerTokenizer(retriever_variant, max_length=max_pieces)
            self._tokenizer_retriever_internal = self._tokenizer_retriever.tokenizer
            token_indexer = PretrainedTransformerIndexer(retriever_variant)
            self._token_indexers_retriever = {'tokens': token_indexer}
        else:
            raise ValueError(
                f"Invalid retriever_variant = {retriever_variant}.\nInvestigate!"
            )

        self._max_pieces = max_pieces
        self._add_prefix = add_prefix
        self._scramble_context = scramble_context
        self._use_context_full = use_context_full
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex
        self._retriever_variant = retriever_variant

    @overrides
    def _read(self, file_path: str):
        instances = self._read_internal(file_path)
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5
        is_done = False

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
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
                    qdep = None
                    if counter == 0:
                        is_done = True
                        break
                    if debug > 0:
                        logger.info(item_json)
                    if self._syntax == "rulebase":
                        text = question['text']
                        q_id = question['id']
                        label = None
                        if 'label' in question:
                            label = 1 if question['label'] else 0
                        qdep = question['meta']['QDep']
                    elif self._syntax == "propositional-meta":
                        text = question[1]['question']
                        q_id = f"{item_id}-{question[0]}"
                        label = question[1]['propAnswer']
                        if label is not None:
                            label = ["False", "True", "Unknown"].index(label)

                    yield self.text_to_instance(
                        item_id=q_id,
                        question_text=text,
                        context=context,
                        label=label,
                        debug=debug,
                        qdep=qdep,
                    )

    @overrides
    def text_to_instance(self,  # type: ignore
        item_id: str,
        question_text: str,
        label: int = None,
        context: str = None,
        debug: int = -1,
        qdep = None,
        qa_only: bool = False,
    ) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        # Tokenize for the qa model
        qa_tokens, segment_ids = self.transformer_features_from_qa(question_text, context)
        qa_field = TextField(qa_tokens, self._token_indexers_qamodel)
        fields['phrase'] = qa_field

        if not qa_only:
            # Tokenize context sentences seperately
            retrieval_listfield = self.listfield_features_from_qa(
                question_text, context, self._tokenizer_retriever
            )
            qa_listfield = self.listfield_features_from_qa(
                question_text, context, self._tokenizer_qamodel
            )
            fields['retrieval'] = ListField(
                [TextField(toks, self._token_indexers_retriever) for toks in retrieval_listfield]
            )
            fields['sentences'] = ListField(
                [TextField(toks, self._token_indexers_qamodel) for toks in qa_listfield]
            )

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "tokens": [x.text for x in qa_tokens],
            "context": context,
            "QDep": qdep,
        }

        if label is not None:
            # We'll assume integer labels don't need indexing
            fields['label'] = LabelField(label, skip_indexing=isinstance(label, int))
            metadata['label'] = label

        if debug > 0:
            logger.info(f"qa_tokens = {qa_tokens}")
            logger.info(f"context = {context}")
            logger.info(f"label = {label}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def transformer_features_from_qa(self, question: str, context: str):
        if self._add_prefix is not None:
            question = self._add_prefix.get("q", "") + question
            context = self._add_prefix.get("c", "") + context
        if context is not None:
            tokens = self._tokenizer_qamodel.tokenize_sentence_pair(question, context)
        else:
            tokens = self._tokenizer_qamodel.tokenize(question)
        segment_ids = [0] * len(tokens)

        return tokens, segment_ids

    def listfield_features_from_qa(self, question: str, context: str, tokenizer):
        ''' Tokenize the context items seperately and return as a list.
        '''
        to_tokenize = (question + (context if context is not None else "")).split('.')[:-1]
        to_tokenize = [toks + '.' for toks in to_tokenize]
        tokens = [tokenizer.tokenize(item) for item in to_tokenize]
        return tokens

    def transformer_indices_from_qa(self, sentences, vocab):
        ''' Convert question + context strings into a batch
            which is ready for the qa model.
        '''
        instances = []
        for question, context in sentences:
            instance = self.text_to_instance(
                item_id="", question_text=question, context=context, qa_only=True
            )
            instance.index_fields(vocab)
            instances.append(instance)

        return allennlp_collate(instances)

    def decode(self, input_id, mode='qa', vocab=None):
        ''' Helper to decode a tokenized sequence.
        '''
        if mode == 'qa':
            func = self._tokenizer_qamodel_internal._convert_id_to_token
        else:
            if vocab:
                func = lambda idx: vocab._index_to_token['tokens'][idx]
            else:
                raise ValueError

        return ' '.join([func(inp) for inp in input_id.tolist()])

    def pad_idx(self, mode):
        if mode == 'qa':
            return self._tokenizer_qamodel_internal.pad_token_id
        elif mode == 'retriever':
            return self._tokenizer_retriever_internal.pad_token_id
        else:
            raise NotImplementedError()
