from typing import Dict, Any
import json
import logging
import random
import re
import numpy as np

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

from .processors import RRProcessor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("proof_reader")
class ProofReader(DatasetReader):
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
        max_depth: int = 5,
    ) -> None:
        super().__init__()
        
        # Initalize retriever tokenizer
        if retriever_variant == 'spacy':
            self._tokenizer = SpacyTokenizer()
            self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        elif 'roberta' in retriever_variant:
            self._tokenizer = PretrainedTransformerTokenizer(retriever_variant, max_length=max_pieces)
            self._tokenizer_internal = self._tokenizer.tokenizer
            token_indexer = PretrainedTransformerIndexer(retriever_variant)
            self._token_indexers = {'tokens': token_indexer}
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
        self._max_depth = max_depth

    @overrides
    def _read(self, file_path: str):
        instances = self._read_internal(file_path)
        return instances

    def _read_internal(self, file_path: str):
        debug = -1

        data_dir = '/'.join(file_path.split('/')[:-1])
        dset = file_path.split('/')[-1].split('.')[0]
        examples = RRProcessor().get_examples(data_dir, dset)

        for example in examples:
            question = example.question.strip()
            qid = example.id
            qdep = example.qdep

            if qdep > self._max_depth:
                continue

            context = [e.strip() + '.' for e in example.context.split('.')]
            support = [con for i, con in zip(example.node_label, context) if i == 1]
            non_support = [con for i, con in zip(example.node_label, context) if i == 0]

            # TODO: account for case when there are disjoint paths for proof
            # TODO: account for multi-hop proofs by adding relevant context to question

            for n, support_item in enumerate(support):
                # Edge case where there are more support items than
                # non-support items
                if len(non_support) == 0:
                    break

                # Ignore NAF node
                # TODO: figure out what to do with NAF nodes
                if support_item == '.':
                    continue

                # Yield positive example
                qid_ = qid + f'-P-{n}'
                yield self.text_to_instance(
                    item_id=qid_,
                    question_text=question,
                    context=support_item,
                    label=1,
                    debug=debug,
                    qdep=qdep,
                )
                # Yield negative examples by randomly selecting
                # a context item which does not contribute to the proof
                try:
                    neg = non_support.pop(np.random.choice(len(non_support)))
                except:
                    raise ValueError
                qid_ = qid + f'-N-{n}'
                yield self.text_to_instance(
                    item_id=qid_,
                    question_text=question,
                    context=neg,
                    label=0,
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
        qa_field = TextField(qa_tokens, self._token_indexers)
        fields['phrase'] = qa_field

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
 
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer) \
            and context is not None:
            tokens = self._tokenizer.tokenize_sentence_pair(question, context)
        else:
            tokens = self._tokenizer.tokenize(question)
            tokens += self._tokenizer.tokenize(context) if context else ''
        segment_ids = [0] * len(tokens)

        return tokens, segment_ids









##### OLD #####

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
