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

from .processors import RRProcessor

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
        pretrained_retriever_model = None,
        topk: int = None,
        concat_q_and_c: bool = None,
        true_samples_only: bool = False,
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
        self._concat = concat_q_and_c if concat_q_and_c is not None else (pretrained_retriever_model is not None)       # TODO
        self._topk = topk
        self._true_samples_only = true_samples_only

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
            if self._true_samples_only:
                # Filter so only positive correct questions and negative
                # incorrect questions are used
                if 'not' in example.question and example.label:
                    continue
                if 'not' not in example.question and not example.label:
                    continue

            if not (example.qlen == '' or int(example.qlen) <= self._topk):
                continue

            yield self.text_to_instance(
                item_id=example.id,
                question_text=example.question.strip(),
                context=example.context,
                label=example.label,
                debug=debug,
                qdep=example.qdep,
                qlen=example.qlen,
                node_label=example.node_label
            )

    @overrides
    def text_to_instance(self,  # type: ignore
        item_id: str,
        question_text: str,
        label: int = None,
        already_retrieved = '',
        context: str = None,
        debug: int = -1,
        qdep: int = None,
        qlen: int = None,
        qa_only: bool = False,
        node_label: list = [],
    ) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        # Tokenize for the qa model
        qa_tokens, _ = self.transformer_features_from_qa(question_text, context)
        qa_field = TextField(qa_tokens, self._token_indexers_qamodel)
        fields['phrase'] = qa_field

        if not qa_only:
            # Tokenize context sentences seperately
            retrieval_listfield = self.listfield_features_from_qa(
                question_text, context, already_retrieved, self._tokenizer_retriever
            )
            qa_listfield = self.listfield_features_from_qa(
                question_text, context, already_retrieved, self._tokenizer_qamodel
            )
            fields['retrieval'] = ListField(
                [TextField(toks, self._token_indexers_retriever) for toks in retrieval_listfield]
            )
            fields['sentences'] = ListField(
                [TextField(toks, self._token_indexers_qamodel) for toks in qa_listfield]
            )
            exact_match = self._get_exact_match(question_text, context)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "tokens": [x.text for x in qa_tokens],
            "context": context,
            "QDep": qdep,
            "node_label": node_label,
            "exact_match": exact_match if not qa_only else None,
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

    def _get_exact_match(self, question, context):
        context_lst = [toks.strip() + '.' for toks in context.split('.')[:-1]]
        exact_match = context_lst.index(question) if question in context_lst else None
        return exact_match

    def listfield_features_from_qa(self, question: str, context: str, already_retrieved, tokenizer):
        ''' Tokenize the context items seperately and return as a list.
        '''
        if self._concat:
            tokens = []
            for toks in context.split('.')[:-1]:
                toks_ = toks.strip() + '.'
                aug_context = (already_retrieved + ' ' + toks_).strip()
                trans_features = self.transformer_features_from_qa(question, aug_context)
                tokens.append(trans_features[0])
        else:
            to_tokenize = (question + (context if context is not None else "")).split('.')[:-1]
            to_tokenize = [toks + '.' for toks in to_tokenize]
            tokens = [tokenizer.tokenize(item) for item in to_tokenize]
        return tokens

    def transformer_indices_from_qa(self, sentences, vocab):
        ''' Convert question + context strings into a batch
            which is ready for the qa model.
        '''
        data = []
        for question, already_retrieved, context in sentences:
            instance = self.text_to_instance(
                item_id = "", 
                question_text = question, 
                context = context, 
                already_retrieved = already_retrieved,
                qa_only = False
            )
            instance.index_fields(vocab)
            data.append(instance)

        return allennlp_collate(data)

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

