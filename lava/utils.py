from datetime import datetime
from functools import wraps
from time import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31


def nested_args_update(dict_overrider, dict_overridden):
    for k,v in dict_overrider.items():
        if isinstance(v, dict):
            if k in dict_overridden:
                dict_overridden[k].update(v)
            else:
                dict_overridden[k] = v
        else:
            dict_overridden.update({k:v})
    return dict_overridden


def dict_to_str(overrides):
    return str(overrides).replace("True", "'True'").replace("False", "'False'").replace("None", "'None'")


def lfilter(func, l):
    return list(filter(func, l))


def lmap(func, l):
    return list(map(func, l))


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def datetime_now():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:"{f.__qualname__}" took: {te-ts:2.4f} sec')
        return result
    return wrap


def batch_lookup(M, idx, vector_output=True):
    """
    Perform batch lookup on matrix M using indices idx.
    :param M: (Variable) [batch_size, seq_len] Each row of M is an independent population.
    :param idx: (Variable) [batch_size, sample_size] Each row of idx is a list of sample indices.
    :param vector_output: If set, return a 1-D vector when sample size is 1.
    :return samples: [batch_size, sample_size] samples[i, j] = M[idx[i, j]]
    """
    batch_size = M.size(0)
    batch_size2, sample_size = idx.size()
    assert(batch_size == batch_size2)

    if sample_size == 1 and vector_output:
        samples = torch.gather(M, 1, idx).view(-1)
    else:
        samples = torch.gather(M, 1, idx)
    pass
    return samples


def safe_log(x):
    return torch.log(x + EPSILON)


def right_pad(x, y, value=0.0):
    if x.size(-1) >= y.size(-1):
        return x
    else:
        output = torch.full_like(y, value)
        output[..., :x.size(-1)] = x
        return output


def correct_legacy_path(path):
    return path.replace('bin/runs/', 'resources/runs/')


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def one_hot(make_as, x):
    return torch.zeros_like(make_as).scatter(1, x.unsqueeze(-1), 1)


def lrange(*args):
    return list(range(*args))


def print_results(answers, scale_n):
    n = scale_n * 100
    for d in sorted(answers.keys()):
        all_score_a = answers[d][0].count(True) / max(len(answers[d][0]), 1)
        last_100_a = answers[d][0][-n:].count(True) / max(len(answers[d][0][-n:]),1)
        all_score_r = answers[d][1].count(True) / max(len(answers[d][1]),1)
        last_100_r = answers[d][1][-n:].count(True) / max(len(answers[d][1][-n:]),1)
        print(f'\nM:\tL: {d}\tAll: {all_score_a:.3f}\tLast {n}: {last_100_a:.2f}\t'
            f'B:\tAll: {all_score_r:.3f}\tLast {n}: {last_100_r:.2f}\tN: {len(answers[d][0])}')


def gs(logits, tau=1):
    ''' Sample using Gumbel Softmax. Ingests raw logits.
    '''
    return F.gumbel_softmax(logits, tau=tau, hard=True, eps=1e-10, dim=-1)
