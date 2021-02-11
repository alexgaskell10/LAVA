import sys
import os
import argparse
import logging
from typing import Any, Optional

from overrides import overrides

from allennlp import __version__
from allennlp.commands.elmo import Elmo
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.find_learning_rate import FindLearningRate
from allennlp.commands.predict import Predict
from allennlp.commands.print_results import PrintResults
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.test_install import TestInstall
from allennlp.commands.train import Train
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules

logger = logging.getLogger(__name__)


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """
    Custom argument parser that will display the default value for an argument
    in the help message.
    """

    _action_defaults_to_ignore = {"help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default: Any) -> bool:
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    @overrides
    def add_argument(self, *args, **kwargs):
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get(
            "action"
        ) not in self._action_defaults_to_ignore and not self._is_empty_default(default):
            description = kwargs.get("help", "")
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


def create_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Creates the argument parser for the main program.
    """
    parser = ArgumentParserWithDefaults(description="Run AllenNLP", prog=prog)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    for subcommand_name in sorted(Subcommand.list_available()):
        subcommand_class = Subcommand.by_name(subcommand_name)
        subcommand = subcommand_class()
        subparser = subcommand.add_subparser(subparsers)
        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )

    return parser


def main(prog: Optional[str] = None) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin (see :mod:`~allennlp.common.plugins`).
    """
    import_plugins()

    if len(sys.argv) == 1:
        os.chdir('ruletaker')
        sys.argv[1:] = ['train', 'allennlp_models/config/tmp.jsonnet', 
            '-s', 'runs/roberta-tiny', '--include-package', 'allennlp_models']
        # sys.argv[1:] = ['evaluate', 'runs/depth-5/model.tar.gz', 'dev', '--output-file', '_results.json', 
        #     '-o', "{'trainer': {'cuda_device': 0}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
        #     '--cuda-device', '0', '--include-package', 'allennlp_models']

        if sys.argv[1] == 'evaluate':
            sys.argv[3] = f"inputs/dataset/rule-reasoning-dataset-V2020.2.4/{sys.argv[2].split('/')[1]}/{sys.argv[3]}.jsonl"
            sys.argv[5] = f"{'/'.join(sys.argv[2].split('/')[:2])}/{sys.argv[3]}/{sys.argv[5]}"

        if 'tmp' in sys.argv[2]:
            os.system(f"rm -rf {sys.argv[4]}")

    parser = create_parser(prog)
    args = parser.parse_args()

    # Hack to use wandb logging
    if sys.argv[1] == 'train' and 'tmp' not in sys.argv[2]:
        import wandb
        wandb.init(project="ruletaker", config=vars(args))
        os.environ['WANDB_LOG'] = 'true'
    else:
        os.environ['WANDB_LOG'] = 'false'

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in args.include_package:
            import_module_and_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


def run():
    main(prog="allennlp")


if __name__ == "__main__":
    run()
