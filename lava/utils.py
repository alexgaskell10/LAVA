import os
import numpy as np
import pickle as pkl
import shutil
import random

import torch
from datetime import datetime
from functools import wraps
from time import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models.archival import CONFIG_NAME

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


def read_pkl(dataset_reader, data_path):
    dset = data_path.split('/')[-1].replace('.jsonl','')
    pkl_file = os.path.join(*data_path.split('/')[:-1], dataset_reader.pkl_file.replace('DSET', dset))
    if os.path.exists(pkl_file) and (dataset_reader.max_instances is None or dataset_reader.max_instances >= 1e2):
        print('Loading pickle file: '+pkl_file)
        with open(pkl_file, 'rb') as f:
            train_data = pkl.load(f)
    else:
        train_data = dataset_reader.read(data_path)
        if dataset_reader.max_instances is None or dataset_reader.max_instances > 1e2:
            with open(pkl_file, 'wb') as f:
                pkl.dump(train_data, f)
    return train_data


def read_all_datasets(
    train_data_path: str,
    dataset_reader,
    validation_dataset_reader = None,
    validation_data_path: str = None,
    test_data_path: str = None,
):
    """
    Reads all datasets (perhaps lazily, if the corresponding dataset readers are lazy) and returns a
    dictionary mapping dataset name ("train", "validation" or "test") to the iterable resulting from
    `reader.read(filename)`.
    """

    logger.info("Reading training data from %s", train_data_path)
    train_data = read_pkl(dataset_reader, train_data_path)

    datasets = {"train": train_data}

    validation_dataset_reader = validation_dataset_reader or dataset_reader

    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = read_pkl(dataset_reader, validation_data_path)
        datasets["validation"] = validation_data

    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = read_pkl(dataset_reader, test_data_path)
        datasets["test"] = test_data

    return datasets


def description_from_metrics(metrics) -> str:
    return (
        ", ".join(
            [
                "%s: %.4f" % (name, value)
                for name, value in metrics.items()
                if name in ['EM', 'loss']
            ]
        )
        + " ||"
    )


def write_records(records, dset, epoch, serialization_dir, outfile=None):
    if outfile is None:
        outfile = os.path.join(serialization_dir, f'{dset}-records_epoch{epoch}.pkl')
    
    logger.info('Writing records to: ' + outfile)
    with open(outfile, 'wb') as f:
        pkl.dump(records, f)


def create_serialization_dir(
    params: Params, serialization_dir: str, recover: bool, force: bool
) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    # Parameters

    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results and logs.
    recover : `bool`
        If `True`, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    force : `bool`
        If `True`, we will overwrite the serialization directory if it already exists.
    """
    if recover and force:
        raise ConfigurationError("Illegal arguments: both force and recover are true.")

    if os.path.exists(serialization_dir) and force:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(
                f"Serialization directory ({serialization_dir}) already exists and is "
                f"not empty. Specify --recover to recover from an existing output folder."
            )

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError(
                "The serialization directory already exists but doesn't "
                "contain a config.json. You probably gave the wrong directory."
            )
        loaded_params = Params.from_file(recovered_config_file)

        # Check whether any of the training configuration differs from the configuration we are
        # resuming.  If so, warn the user that training may fail.
        fail = False
        flat_params = params.as_flat_dict()
        flat_loaded = loaded_params.as_flat_dict()
        for key in flat_params.keys() - flat_loaded.keys():
            logger.error(
                f"Key '{key}' found in training configuration but not in the serialization "
                f"directory we're recovering from."
            )
            # fail = True
        for key in flat_loaded.keys() - flat_params.keys():
            logger.error(
                f"Key '{key}' found in the serialization directory we're recovering from "
                f"but not in the training config."
            )
            # fail = True
        # for key in flat_params.keys():
        #     if flat_params.get(key) != flat_loaded.get(key):
        #         logger.error(
        #             f"Value for '{key}' in training configuration does not match that the value in "
        #             f"the serialization directory we're recovering from: "
        #             f"{flat_params[key]} != {flat_loaded[key]}"
        #         )
                # fail = True
        if fail:
            raise ConfigurationError(
                "Training configuration does not match the configuration we're recovering from."
            )
    else:
        if recover:
            raise ConfigurationError(
                f"--recover specified but serialization_dir ({serialization_dir}) "
                "does not exist.  There is nothing to recover from."
            )
        os.makedirs(serialization_dir, exist_ok=True)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)