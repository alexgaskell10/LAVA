import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from graphviz import Digraph

import cProfile

import logging
logger = logging.getLogger(__name__)

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:"{f.__qualname__}" took: {te-ts:2.4f} sec')
        return result
    return wrap


# @timing
# def func(meta_records):
#     return call_theorem_prover_from_lst(instances=meta_records)

# @timing
# def func1(meta_records):
#     processes = []
#     for i in range(0,64,24):
#         p = multiprocessing.Process(target=call_theorem_prover_from_lst, args=(meta_records[i:i+24],))
#         processes.append(p)
#         p.start()

#     for process in processes:
#         process.join()
    
#     return process


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


def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style="filled", shape="box", align="left", fontsize="12", ranksep="0.1", height="0.2")
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                var = var.grad_fn
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)
    return dot


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def one_hot(make_as, x):
    return torch.zeros_like(make_as).scatter(1, x.unsqueeze(-1), 1)


def lfilter(*args):
    return list(filter(*args))


def lmap(*args):
    return list(map(*args))


def flatten_list(l):
    return [item for sublist in l for item in sublist]


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
