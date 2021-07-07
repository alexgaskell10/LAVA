import numpy as np

def lfilter(*args):
    return list(filter(*args))


def lmap(*args):
    return list(map(*args))


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def duplicate_list(lst: list, n: int):
    ''' [1,2,3] --> [1,1,2,2,3,3] '''
    return np.concatenate([([i]*n) for i in lst], axis=0).tolist()


def lrange(*args):
    return list(range(*args))