import os
import numpy as np
import pickle as pkl
import logging
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
    if os.path.exists(pkl_file) and (dataset_reader.max_instances is None or dataset_reader.max_instances > 1e3):
        print('Loading pickle file: '+pkl_file)
        with open(pkl_file, 'rb') as f:
            train_data = pkl.load(f)
    else:
        train_data = dataset_reader.read(data_path)
        if dataset_reader.max_instances is None:
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
