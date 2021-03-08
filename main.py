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
        ## RT
        # sys.argv[1:] = ['train', 'ruletaker/allennlp_models/config/tmp.jsonnet', 
        #     '-s', 'ruletaker/runs/t16', '--include-package', 'ruletaker.allennlp_models']
        ## Retrieval + ruletaker
        # sys.argv[1:] = ['custom_train', 'bin/config/spacy_tmp.jsonnet', # {'bin/config/tmp_new.jsonnet', 'bin/config/spacy_retriever.jsonnet', 'bin/config/pretrain_retriever_tmp.jsonnet'},
        #     '-s', 'bin/runs/pretrain_retriever/tmp', '--include-package', 'ruletaker.allennlp_models']
        # # Pretrain retriever
        # sys.argv[1:] = ['train', 'bin/config/pretrain_retriever.jsonnet', # {'bin/config/tmp_new.jsonnet', 'bin/config/spacy_retriever.jsonnet', 'bin/config/pretrain_retriever_tmp.jsonnet'},
        #     '-s', 'bin/runs/pretrain_retriever/rb-base', '--include-package', 'ruletaker.allennlp_models']
        # Pretrained retrieval + ruletaker
        # sys.argv[1:] = ['custom_train', 'bin/config/trained_roberta_retriever.jsonnet', # {'bin/config/tmp_new.jsonnet', 'bin/config/spacy_retriever.jsonnet', 'bin/config/pretrain_retriever_tmp.jsonnet'},
        #     '-s', 'bin/runs/pretrained_retriever_ruletaker/roberta/tmp', '--include-package', 'ruletaker.allennlp_models']
        # RL/GS
        sys.argv[1:] = ['custom_train', 'bin/config/gumbel_softmax_tmp.jsonnet', # {'bin/config/tmp_new.jsonnet', 'bin/config/spacy_retriever.jsonnet', 'bin/config/pretrain_retriever_tmp.jsonnet'},
            '-s', 'bin/runs/pretrained_retriever_ruletaker/roberta/tmp1', '--include-package', 'ruletaker.allennlp_models']

        # sys.argv[1:] = ['evaluate', 'ruletaker/runs/depth-5-base/model.tar.gz', 'dev', '--output-file', '_results.json', 
        #     '-o', "{'trainer': {'cuda_device': 0}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
        #     '--cuda-device', '0', '--include-package', 'ruletaker.allennlp_models']

        if sys.argv[1] == 'evaluate':
            dset = re.search(r'/(depth-.+?)[/-]', sys.argv[2]).group(1)
            sys.argv[3] = f"ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/{dset}/{sys.argv[3]}.jsonl"
            sys.argv[5] = f"{'/'.join(sys.argv[2].split('/')[:3])}/{sys.argv[3].strip('.jsonl') + sys.argv[5]}"

        if 'tmp' in sys.argv[2] or 'tmp' in sys.argv[4]:
            if os.path.isdir(sys.argv[4]):
                os.system(f"rm -rf {sys.argv[4]}")

    parser = create_parser(prog)
    args = parser.parse_args()

    # Hack to use wandb logging
    if 'train' in sys.argv[1] and 'tmp' not in sys.argv[2]:
        import wandb
        if 'pretrain_retriever' in sys.argv[2]:
            project = "re-re_pretrain-ret"  
        elif 'gumbel_softmax' in sys.argv[2]:
            project = "re-re_gumbel-softmax"
        else:
            project = "re-re"

        wandb.init(project=project, config=vars(args))
        os.environ['WANDB_LOG'] = 'true'
    else:
        os.environ['WANDB_LOG'] = 'false'

    for package_name in args.include_package:
        import_module_and_submodules(package_name)
    args.func(args)


if __name__ == '__main__':
    main()