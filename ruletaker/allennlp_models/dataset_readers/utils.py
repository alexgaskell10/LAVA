def lfilter(func, l):
    return list(filter(func, l))


def lmap(func, l):
    return list(map(func, l))


def flatten_list(l):
    return [item for sublist in l for item in sublist]