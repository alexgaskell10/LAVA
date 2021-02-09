import sys
import os
import argparse
import logging
import re
from typing import Any, Optional

from overrides import overrides

from allennlp.common.util import import_module_and_submodules
from allennlp.common.plugins import import_plugins
from allennlp.commands import create_parser
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import dump_metrics, prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
# from allennlp.training.util import evaluate
from allennlp.commands.train import TrainModel
from allennlp.training import util as training_util

from ruletaker.allennlp_models.train.custom_train import CustomTrain

logger = logging.getLogger(__name__)


def main(prog: Optional[str] = None) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin (see :mod:`~allennlp.common.plugins`).
    """
    import_plugins()

    if len(sys.argv) == 1:
        # sys.argv[1:] = ['train', 'ruletaker/allennlp_models/config/tmp.jsonnet', 
        #     '-s', 'ruletaker/runs/t16', '--include-package', 'ruletaker.allennlp_models']
        sys.argv[1:] = ['custom_train', 'bin/config/tmp.jsonnet', # {'bin/config/tmp_new.jsonnet', 'bin/config/spacy_retriever.jsonnet'},
            '-s', 'bin/runs/tmp', '--include-package', 'ruletaker.allennlp_models']
        # sys.argv[1:] = ['evaluate', 'ruletaker/runs/depth-5-base/model.tar.gz', 'dev', '--output-file', '_results.json', 
        #     '-o', "{'trainer': {'cuda_device': 0}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
        #     '--cuda-device', '0', '--include-package', 'ruletaker.allennlp_models']

        if sys.argv[1] == 'evaluate':
            dset = re.search(r'/(depth-.+?)[/-]', sys.argv[2]).group(1)
            sys.argv[3] = f"ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/{dset}/{sys.argv[3]}.jsonl"
            sys.argv[5] = f"{'/'.join(sys.argv[2].split('/')[:3])}/{sys.argv[3].strip('.jsonl') + sys.argv[5]}"

        if 'tmp' in sys.argv[2]:
            if os.path.isdir(sys.argv[4]):
                os.system(f"rm -rf {sys.argv[4]}")

    parser = create_parser(prog)
    args = parser.parse_args()

    # Hack to use wandb logging
    if 'train' in sys.argv[1] and 'tmp' not in sys.argv[2]:
        import wandb
        wandb.init(project="re-re", config=vars(args))
        os.environ['WANDB_LOG'] = 'true'
    else:
        os.environ['WANDB_LOG'] = 'false'

    for package_name in args.include_package:
        import_module_and_submodules(package_name)
    args.func(args)

def run(args):
    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model

    # TODO reconcile config with args
    args.model = model


def run_eval(args):
    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model

    # Manually adjust config and args
    config = adjust_config(config)
    args = adjust_args(args)

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    embedding_sources = (
        json.loads(args.embedding_sources_mapping) if args.embedding_sources_mapping else {}
    )

    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(instances=instances)
        model.extend_embedder_vocab(embedding_sources)

    instances.index_with(model.vocab)
    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if args.batch_size:
        data_loader_params["batch_size"] = args.batch_size
    data_loader = DataLoader.from_params(dataset=instances, params=data_loader_params)

    # metrics = evaluate(model, data_loader, args.cuda_device, args.batch_weight_key)

    # logger.info("Finished evaluating.")

    # dump_metrics(args.output_file, metrics, log=False)


if __name__ == '__main__':
    main()