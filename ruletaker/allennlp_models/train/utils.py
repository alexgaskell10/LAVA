import os
import numpy as np
import pickle as pkl
import datetime
import logging
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model
from allennlp.nn import util as nn_util


logger = logging.getLogger(__name__)


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


def write_records(records, dset, epoch, serialization_dir):
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
