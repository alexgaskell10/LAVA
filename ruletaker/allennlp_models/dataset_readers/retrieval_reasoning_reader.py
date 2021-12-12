from typing import Dict, Any
import random, re, os, json, logging, pickle
from copy import deepcopy

import numpy as np
from torch import Tensor
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field, TextField, LabelField, MetadataField, SequenceLabelField,
    ListField, ArrayField
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
        longest_proof: int = 100,
        shortest_proof: int = 1,
        concat_q_and_c: bool = True,
        true_samples_only: bool = False,
        add_NAF: bool = False,
        one_proof: bool = False,
        word_overlap_scores: bool = False,
        max_instances: int = None,
    ):
        max_instances = None if not max_instances or max_instances == -1 else max_instances
        super().__init__(cache_directory=None, max_instances=max_instances)
        
        # Init reasoning tokenizer
        self._tokenizer_qamodel = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces)
        self._tokenizer_qamodel_internal = self._tokenizer_qamodel.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers_qamodel = {'tokens': token_indexer}
        
        # Initalize retriever tokenizer
        if 'roberta' in retriever_variant:
            self._tokenizer_retriever = PretrainedTransformerTokenizer(retriever_variant, max_length=max_pieces)
            self._tokenizer_retriever_internal = self._tokenizer_retriever.tokenizer
            token_indexer = PretrainedTransformerIndexer(retriever_variant)
            self._token_indexers_retriever = {'tokens': token_indexer}
        else:
            raise ValueError(f"Invalid retriever_variant = {retriever_variant}.\nInvestigate!")

        if word_overlap_scores:
            from rouge_score.rouge_scorer import RougeScorer
            self.rouge_scorer = RougeScorer(["rouge1"])
            self._word_overlap_scores_lst = []

        self._max_pieces = max_pieces
        self._add_prefix = add_prefix
        self._scramble_context = scramble_context
        self._use_context_full = use_context_full
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex
        self._retriever_variant = retriever_variant
        self._concat = concat_q_and_c if concat_q_and_c is not None else (pretrained_retriever_model is not None)       # TODO
        self._longest = longest_proof
        self._shortest = shortest_proof
        self._true_samples_only = true_samples_only
        self._add_NAF = add_NAF
        self._one_proof = one_proof     # Limits the dataset to questions with a single proof. Cuts dataset size by c. 12%
        self._word_overlap_scores = word_overlap_scores
        tok = type(self).__name__
        self.pkl_file = f'{tok}_{self._max_pieces}_{self._shortest}_{self._longest}_{int(self._true_samples_only)}_{int(self._add_NAF)}_{int(self._one_proof)}_{int(self._word_overlap_scores)}_{max_instances}_DSET.pkl'

    @overrides
    def _read(self, file_path: str):
        instances = self._read_internal(file_path)
        return instances

    def _read_internal(self, file_path: str):
        debug = -1
        debug_num = -1 #100

        data_dir = '/'.join(file_path.split('/')[:-1])
        dset = file_path.split('/')[-1].split('.')[0]
        examples = RRProcessor().get_examples(data_dir, dset, debug_num=self.max_instances, one_proof=self._one_proof)

        for example in examples:
            example.qlen = sum(example.node_label)

            if self._true_samples_only:
                # Filter so only positive correct questions and negative
                # incorrect questions are used
                if 'not' in example.question and example.label:
                    continue
                if 'not' in example.question:
                    continue
                if 'not' not in example.question and not example.label:
                    continue
                if example.node_label[-1] == 1:
                    continue

            if not (self._shortest <= int(example.qlen) <= self._longest):
                continue

            yield self.text_to_instance(
                item_id=example.id,
                question_text=example.question.strip(),
                context=example.context,
                label=example.label,
                debug=debug,
                qdep=example.qdep,
                qlen=example.qlen,
                node_label=example.node_label,
                meta_record=example.meta_record,
                sentence_scramble=example.sentence_scramble,
            )

    @overrides
    def text_to_instance(self,
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
        initial_tokenization: bool = True,
        disable: bool = False,
        meta_record: dict = {},
        sentence_scramble: list = [],
    ) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        if self._add_NAF and initial_tokenization:
            # Add NAF node to allow model to attend to no sentence.
            context += ' NAF. '

        # Tokenize for the qa model
        qa_tokens, _ = self.transformer_features_from_qa(question_text, context)
        qa_field = TextField(qa_tokens, self._token_indexers_qamodel)
        fields['phrase'] = qa_field

        if not disable:
            meta_record = self.append_flipped_question(item_id, meta_record)
            n_facts = len(meta_record['triples'])
            n_rules = len(meta_record['rules'])
            fact_idx = sorted([sentence_scramble.index(n+1) for n in range(n_facts)])
            rule_idx = sorted([sentence_scramble.index(n+1) for n in range(n_facts, n_facts + n_rules)])
            assert (len(fact_idx) == n_facts) and (len(rule_idx) == n_rules)

        if True:
            exact_match = self._get_exact_match(question_text, context)

        if self._word_overlap_scores and not disable:
            scores = [self.rouge_scorer.score(question_text, c)["rouge1"].fmeasure for c in context.split('.')[:-1]]
            fields["word_overlap_scores"] = ArrayField(np.array(scores))
            self._word_overlap_scores_lst.extend(scores)

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "tokens": [x.text for x in qa_tokens],
            "context": context,
            "QDep": qdep,
            "node_label": node_label,
            "exact_match": exact_match if not qa_only else None,
            "QLen": qlen,
            "meta_record": meta_record,
            "sentence_scramble": sentence_scramble,
            "fact_indices": fact_idx if not disable else None,
            "rule_indices": rule_idx if not disable else None,
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

    def append_flipped_question(self, item_id, meta_record_):
        meta_record = deepcopy(meta_record_)
        qid = 'Q'+item_id.split('-')[-1]
        ques_record = meta_record['questions'][qid]
        ques = ques_record['question']
        ques_rep = ques_record['representation'].lstrip('("').rstrip('")').split('" "')
        pred = ques_rep[1]
        polarity = ques_rep[-1]
        anti_pred = 'is not' if pred == 'is' else f'does not {pred.rstrip("s")}'      # needs --> does not need
        if polarity == '+':
            assert pred in ques
            flipped_ques = ques.replace(pred, anti_pred)
        elif polarity == '-':
            assert anti_pred in ques
            flipped_ques = ques.replace(anti_pred, pred)
        meta_record['questions'][qid]['flipped_ques'] = flipped_ques
        return meta_record

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
        exact_match = context_lst.index(question) if question in context_lst else -1
        return exact_match

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

    def encode_batch(self, sentences, vocab, disable, device=False):
        ''' Convert question + context strings into a batch
            which is ready for the qa model.
        '''
        data = []
        for question, already_retrieved, label in sentences:
            instance=self.text_to_instance(
                item_id="", 
                question_text=question, 
                context=already_retrieved,
                qa_only=True,
                label=label.item() if isinstance(label, Tensor) else 0,
                initial_tokenization=False,
                disable=disable
            )
            instance.index_fields(vocab)
            data.append(instance)

        if device:
            return self.move(allennlp_collate(data), device)
        else:
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
            raise NotImplementedError

    def encode_token(self, tok, mode):
        if mode == 'qa':
            return self._tokenizer_qamodel.tokenizer.encoder[tok]
        elif mode == 'retriever':
            return self._tokenizer_retriever.tokenizer.encoder[tok]
        else:
            raise NotImplementedError

    def move(self, d: dict, device) -> dict:
        for k in d:
            if isinstance(d[k], dict):
                d[k] = self.move(d[k], device)
            elif isinstance(d[k], Tensor):
                d[k] = d[k].to(device=device, non_blocking=True)
        return d       
            
    def tok(self, x):
        return [tok.lstrip('Ġ') for tok in self._tokenizer_qamodel.tokenizer.tokenize(x) if tok!='Ġ']
