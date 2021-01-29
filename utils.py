import os
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from transformers.tokenization_utils_base import ExplicitEnum
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score

class Features:
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, orig_text):
        self.input_ids = input_ids
        self.attention_mask = input_mask
        self.token_type_ids = segment_ids
        self.labels = label_id
        self.orig_text = orig_text


def compute_metrics(outputs):
    ys = outputs.label_ids
    preds = [0 if p1>p2 else 1 for p1, p2 in outputs.predictions]
    return {
        'accuracy': accuracy_score(ys, preds),
        # 'f1': f1_score(ys, preds),
    }


def collate_fn(inputs):
    collated = {}
    for attribute in inputs[0].__dict__.keys():
        if attribute != 'orig_text':
            collated[attribute] = torch.tensor([getattr(i, attribute) for i in inputs])
        else:
            collated[attribute] = [getattr(i, attribute) for i in inputs]
    return collated        


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
        Code taken from https://github.com/yakazimir/semantic_fragments
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_features(samples, max_seq_length, tokenizer):
    """ Convert data samples into tokenized model features.
        Code taken from https://github.com/yakazimir/semantic_fragments
    """
    label_map = {True: 1, False: 0}
    features = []
    for ex_index, sample in tqdm(enumerate(samples)):
        tokens_a = tokenizer.tokenize(sample[1])
        tokens_b = tokenizer.tokenize(sample[2])
        label_id = label_map[sample[0]]
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                Features(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            orig_text=tokens_a + tokens_b))
    return features


def dir_empty_or_nonexistent(args):
    return not (os.path.exists(args.output_dir) and os.listdir(args.output_dir))

