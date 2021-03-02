import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31

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
    if x.size(-1) < y.size(-1):
        return x
    else:
        output = torch.full_like(y, value)
        output[..., :x.size(-1)] = x
        return output