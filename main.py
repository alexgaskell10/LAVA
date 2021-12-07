import sys
import os
import argparse
import logging
import re
from typing import Any, Optional
from datetime import datetime

from overrides import overrides

from allennlp.common.util import import_module_and_submodules
from allennlp.common.plugins import import_plugins
from allennlp.commands import create_parser
from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import dump_metrics, prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.commands.train import TrainModel
from allennlp.training import util as training_util

from ruletaker.allennlp_models.train.custom_train import CustomTrain

logger = logging.getLogger(__name__)


def datetime_now():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def main(prog: Optional[str] = None) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin (see :mod:`~allennlp.common.plugins`).
    """
    import_plugins()

    outdir_adv = 'bin/runs/adversarial/'+datetime_now()
    outdir_rt = 'ruletaker/runs/'+datetime_now()

    cmds = {
        # "ruletaker_train_original": ['train', 'ruletaker/allennlp_models/config/rulereasoning_config.jsonnet', '-s', outdir_rt, '--include-package', 'ruletaker.allennlp_models'],
        "ruletaker_train_original": ['train', 'bin/config/ruletaker/rulereasoning_config.jsonnet', '-s', outdir_rt, '--include-package', 'ruletaker.allennlp_models'],
        "ruletaker_adv_training": ['re_train', 'ruletaker/allennlp_models/config/ruletaker_adv_retraining.jsonnet', '-s', outdir_rt, '--include-package', 'ruletaker.allennlp_models'],
        "adversarial_dataset_generation": ['custom_train', 'bin/config/adv_tmp.jsonnet', '-s', outdir_adv, '--include-package', 'ruletaker.allennlp_models'],
        "ruletaker_eval_original": [
            'evaluate', 'ruletaker/runs/depth-5/model.tar.gz', 'dev', '--output-file', '_results.json', 
            '-o', "{'trainer': {'cuda_device': 3}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
            '--cuda-device', '3', '--include-package', 'ruletaker.allennlp_models'
        ]
    }

    if len(sys.argv) == 2:
        sys.argv[1:] = cmds[sys.argv.pop(1)]

        if sys.argv[1] == 'evaluate':
            dset = re.search(r'/(depth-.+?)[/-]', sys.argv[2]).group(1)
            sys.argv[3] = f"ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/{dset}/{sys.argv[3]}.jsonl"
            # sys.argv[5] = f"{'/'.join(sys.argv[2].split('/')[:3])}/{sys.argv[3].strip('.jsonl') + sys.argv[5]}"
            sys.argv[5] = f"{'/'.join(sys.argv[2].split('/')[:3])}/{sys.argv[3].split('/')[-1].strip('.jsonl') + sys.argv[5]}"

        if 'tmp' in sys.argv[2] or 'tmp' in sys.argv[4]:
            while os.path.isdir(sys.argv[4]):
                if os.path.isdir(sys.argv[4]):
                    os.system(f"rm -rf {sys.argv[4]}")
                if os.path.isdir(sys.argv[4]):
                    sys.argv[4] += '.1'

    parser = create_parser(prog)
    args = parser.parse_args()

    logging.info('Args loaded')

    # Hack to use wandb logging
    # if False:
    if 'train' in sys.argv[1]:# and 'tmp' not in sys.argv[2]:
        import wandb

        if 'pretrain_retriever' in sys.argv[2]:
            project = "re-re_pretrain-ret"  
        elif 'gumbel_softmax' in sys.argv[2]:
            project = "re-re_gumbel-softmax"
        elif 'ruletaker' in sys.argv[2]:
            project = "ruletaker"
        elif 'adv' in sys.argv[2]:
            project = "adversarial"
        else:
            project = "re-re"

        from _jsonnet import evaluate_file
        import json
        file_dict = json.loads(evaluate_file(args.param_path))

        wandb.init(project=project, config={**vars(args), **file_dict})
        os.environ['WANDB_LOG'] = 'true'
        args.wandb_name = wandb.run.name
    else:
        os.environ['WANDB_LOG'] = 'false'

    for package_name in args.include_package:
        import_module_and_submodules(package_name)
    args.func(args)


if __name__ == '__main__':
    main()