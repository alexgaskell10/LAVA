"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--batch-size BATCH_SIZE]
                             [--batch-weight-key BATCH_WEIGHT_KEY]
                             [--extend-vocab]
                             [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
      archive_file          path to an archived trained model
      input_file            path to the file containing the evaluation data

    optional arguments:
      -h, --help            show this help message and exit
      --output-file OUTPUT_FILE
                            path to output file
      --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
      --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --batch-size BATCH_SIZE
                            If non-empty, the batch size to use during evaluation.
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embedding pretrained-file used during
                            training. If not passed, and embedding needs to be
                            extended, we will try to use the original file paths
                            used during training. If they are not available we
                            will use random vectors for embedding extension.
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
import argparse
import json
import logging
from typing import Any, Dict

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import dump_metrics, prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate

from allennlp.common import Params
from torch.types import Storage

from .utils import write_records

logger = logging.getLogger(__name__)


@Subcommand.register("custom_evaluate")
class Evaluate(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate the specified model + dataset"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate the specified model + dataset."
        )

        subparser.add_argument("archive_file", type=str, help="path to an archived trained model")

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the evaluation data"
        )

        subparser.add_argument("--output-file", type=str, help="path to output file")

        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.add_argument(
            "--batch-size", type=int, help="If non-empty, the batch size to use during evaluation."
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
            "extend your vocabulary. If pretrained-file was used to initialize "
            "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=str,
            default="",
            help="a JSON dict defining mapping from embedding module path to embedding "
            "pretrained-file used during training. If not passed, and embedding needs to be "
            "extended, we will try to use the original file paths used during training. If "
            "they are not available we will use random vectors for embedding extension.",
        )

        subparser.add_argument(
            "--fresh-init",
            action='store_true',
            help="will evaluate using a freshly initialized model",
        )
        subparser.add_argument(
            "--overrides_file",
            help="manually pass the path to overrides so they can be loaded later",
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    if not args.fresh_init:
        # Load from archive
        archive = load_archive(args.archive_file, args.cuda_device, None, args.overrides, args.weights_file)
        config = archive.config
        prepare_environment(config)
        model = archive.model
        model.eval()
        config = Params(json.loads(json.dumps(config.as_dict()).replace('"True"', "true").replace('"False"', "false").replace('"None"', "none")))

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

    else:
        config = Params(args.config)
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

        archive = load_archive(config['model']['ruletaker_archive'], args.cuda_device, None, None, None)
        victim = archive.model
        Model = config.pop('model_class')
        model = Model(qa_model=victim, dataset_reader=dataset_reader, **config['model'])
        model.vocab = victim.vocab
        model.eval()
        model.qa_model.train()
        model = model.to(victim._classifier.bias.device)


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

    metrics = evaluate(model, data_loader, args.cuda_device, args.batch_weight_key)

    logger.info("Finished evaluating.")

    if 'predictions' in metrics:
        # outdir = '/'.join(args.output_file.split('/')[:-1])
        # dset = 'test' if 'test' in evaluation_data_path.split('/')[-1] else 'val'
        outfile = args.output_file.replace('.json', '-records.pkl')
        write_records(metrics.pop('predictions'), None, None, None, outfile)
    dump_metrics(args.output_file, metrics, log=False)

    return metrics
