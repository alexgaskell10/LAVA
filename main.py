import sys
import os
import logging
import _jsonnet, json
from typing import Any, Optional
from datetime import datetime
import pkgutil

from allennlp.common.util import import_module_and_submodules
from allennlp.common.plugins import import_plugins
from allennlp.commands import create_parser

import ruletaker.allennlp_models

logger = logging.getLogger(__name__)


def datetime_now():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def main(prog: Optional[str] = None) -> None:
    import_plugins()

    outdir_adv = 'bin/runs/adversarial/'+datetime_now()
    outdir_rt = 'bin/runs/ruletaker/'+datetime_now()
    outdir_random_bl = 'bin/runs/baselines/random_adversarial/'+datetime_now()

    cmds = {
        "ruletaker_train_original": ['ruletaker_train_original', 'bin/config/ruletaker/rulereasoning_config.jsonnet', '-s', outdir_rt, '--include-package', 'ruletaker.allennlp_models'],
        "ruletaker_adv_training": ['ruletaker_adv_training', 'bin/config/ruletaker/ruletaker_adv_retraining.jsonnet', '-s', outdir_rt, '--include-package', 'ruletaker.allennlp_models'],
        "adversarial_dataset_generation": ['adversarial_dataset_generation', 'bin/config/attacker/tmp.jsonnet', '-s', outdir_adv, '--include-package', 'ruletaker.allennlp_models'],
        "ruletaker_adv_training_test": ['ruletaker_adv_training_test', 
            'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base_retrain/model.tar.gz', 'test', '--output-file', '_results.json', 
            '--overrides_file', 'bin/config/ruletaker/ruletaker_adv_retraining_test_2021-12-12_19-08-47.jsonnet',\
            '--cuda-device', '9', '--include-package', 'ruletaker.allennlp_models'
        ],
        "ruletaker_eval_original": ['ruletaker_eval_original',
            'bin/runs/ruletaker/depth-5/model.tar.gz', 'dev', '--output-file', '_results.json', 
            '-o', "{'trainer': {'cuda_device': 3}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
            '--cuda-device', '3', '--include-package', 'ruletaker.allennlp_models'
        ],
        "ruletaker_test_original": ['ruletaker_test_original',
            'bin/runs/ruletaker/depth-5/model.tar.gz', 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl', 
            '--output-file', '_results.json', 
            '-o', "{'trainer': {'cuda_device': 9}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
            '--cuda-device', '9', '--include-package', 'ruletaker.allennlp_models'
        ],
        "baseline_test": ['baseline_test',
            'bin/runs/ruletaker/depth-5/model.tar.gz', 'bin/runs/baselines/textfooler/2021-12-21_18-12-03_reevaled.pkl', 
            '--output-file', 'bin/runs/baselines/textfooler/2021-12-21_18-12-03_reevaled_results_abc.pkl', 
            '--overrides_file', 'bin/config/baselines/transferability/config.jsonnet',
            '--cuda-device', '9', 
            '--include-package', 'ruletaker.allennlp_models'
        ],
        "adversarial_dataset_generation_test": ['adversarial_dataset_generation_test',
            'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/model.tar.gz', 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl', 
            '--output-file', '_results.json',
            '--overrides_file', 'bin/config/attacker/test_config_2021-12-12_17-38-38.jsonnet',
            '--cuda-device', '9', 
            '--include-package', 'ruletaker.allennlp_models'
        ],
        "transferability": ['transferability',
            'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/model.tar.gz', 
            'bin/runs/baselines/hotflip//2021-12-16_11-00-14_reevaled_bu.pkl', 
            '--output-file', 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/transferability_results_rb_lg--rb_base_hotflip.json', 
            '--overrides_file', 'bin/config/transferability/config_2021-12-20_20-39-00.jsonnet',
            '--cuda-device', '8', '--include-package', 'ruletaker.allennlp_models'
        ],
        "adversarial_random_benchmark": ["adversarial_random_benchmark",
            '', 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
            '--output-file', f'{outdir_random_bl}_results.json',
            '--overrides_file', 'bin/config/baselines/adversarial_benchmark/config.jsonnet',
            '--cuda-device', '8',
            '--fresh-init',
            '--include-package', 'ruletaker.allennlp_models'
        ],
    }

    cmd_map = {
        "ruletaker_train_original":'train',
        "ruletaker_adv_training": 'train',
        "adversarial_dataset_generation": 'custom_train',
        "ruletaker_adv_training_test": 'custom_reevaluate',
        "ruletaker_eval_original": 'evaluate',
        "ruletaker_test_original": 'evaluate',
        "baseline_test": 'custom_evaluate',
        "transferability": 'custom_evaluate',
        "adversarial_dataset_generation_test": 'custom_evaluate',
        "adversarial_random_benchmark": 'custom_evaluate',
    }

    if len(sys.argv) == 2:
        shortcut_launch(cmds)

    cmd = sys.argv[1]
    sys.argv[1] = cmd_map[cmd]

    if cmd == 'adversarial_dataset_generation_test':
        launch_adversarial_dataset_generation_test()
    if cmd == 'ruletaker_adv_training_test':
        launch_ruletaker_adv_training_test()
    if cmd == 'transferability':
        launch_transferability()
    if cmd == 'baseline_test':
        launch_baseline_test()

    parser = create_parser(prog)
    args = parser.parse_args()

    if cmd == 'adversarial_random_benchmark':
        args = launch_adversarial_random_benchmark(args)

    logging.info('Args loaded')

    # Hack to use wandb logging
    if False:
    # if 'train' in sys.argv[1] and pkgutil.find_loader('wandb') is not None:
        import wandb

        if 'pretrain_retriever' in sys.argv[2]:
            project = "re-re_pretrain-ret"  
        elif 'gumbel_softmax' in sys.argv[2]:
            project = "re-re_gumbel-softmax"
        elif 'ruletaker' in sys.argv[2]:
            project = "ruletaker"
            os.environ['WANDB_LOG'] = 'true'
        elif 'attacker' in sys.argv[2]:
            project = "adversarial"
        else:
            project = "re-re"

        from _jsonnet import evaluate_file
        import json
        file_dict = json.loads(evaluate_file(args.param_path))

        wandb.init(project=project, config={**vars(args), **file_dict})
        os.environ['WANDB_LOG_1'] = 'true'
        args.wandb_name = wandb.run.name
    else:
        os.environ['WANDB_LOG'] = 'false'

    for package_name in args.include_package:
        import_module_and_submodules(package_name)
    args.func(args)


def shortcut_launch(cmds):
    sys.argv[1:] = cmds[sys.argv.pop(1)]

    if sys.argv[1].endswith('evaluate') or sys.argv[1].endswith('test'):
        # sys.argv[3] = f"ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/{sys.argv[3]}.jsonl"
        sys.argv[5] = f"{'/'.join(sys.argv[2].split('/')[:4])}/{sys.argv[3].split('/')[-1].strip('.jsonl') + sys.argv[5]}"

    if 'tmp' in sys.argv[2] or 'tmp' in sys.argv[4]:
        while os.path.isdir(sys.argv[4]):
            if os.path.isdir(sys.argv[4]):
                os.system(f"rm -rf {sys.argv[4]}")
            if os.path.isdir(sys.argv[4]):
                sys.argv[4] += '.1'


def launch_adversarial_dataset_generation_test():
    ix = sys.argv.index('--overrides_file')
    sys.argv.pop(ix)                        # Pop flag
    overrides_file = sys.argv.pop(ix)       # And pop path

    file_overrides = json.loads(_jsonnet.evaluate_file(overrides_file))
    model_overrides = json.load(open(sys.argv[2].replace('model.tar.gz', 'config.json'), 'r'))
    for k,v in file_overrides.items():
        if isinstance(v, dict):
            if k in model_overrides:
                model_overrides[k].update(v)
            else:
                model_overrides[k] = v
        else:
            model_overrides.update({k:v})

    model_overrides['model'] = model_overrides.pop('retrieval_reasoning_model')

    sys.argv.extend(['-o', str(model_overrides).replace("True", "'True'").replace("False", "'False'").replace("None", "'None'")])


def launch_ruletaker_adv_training_test():
    ix = sys.argv.index('--overrides_file')
    sys.argv.pop(ix)                        # Pop flag
    overrides_file = sys.argv.pop(ix)       # And pop path

    file_overrides = json.loads(_jsonnet.evaluate_file(overrides_file))
    model_overrides = json.load(open(sys.argv[2].replace('model.tar.gz', 'config.json'), 'r'))
    for k,v in file_overrides.items():
        if isinstance(v, dict):
            if k in model_overrides:
                model_overrides[k].update(v)
            else:
                model_overrides[k] = v
        else:
            model_overrides.update({k:v})

    model_overrides['dataset_reader'].pop('adversarial_examples_path', None)
    sys.argv.extend(['-o', str(model_overrides).replace("True", "'True'").replace("False", "'False'").replace("None", "'None'")])


def launch_transferability():
    ix = sys.argv.index('--overrides_file')
    sys.argv.pop(ix)                        # Pop flag
    overrides_file = sys.argv.pop(ix)       # And pop path

    file_overrides = json.loads(_jsonnet.evaluate_file(overrides_file))
    model_overrides = json.load(open(sys.argv[2].replace('model.tar.gz', 'config.json'), 'r'))
    for k,v in file_overrides.items():
        if isinstance(v, dict):
            if k in model_overrides:
                model_overrides[k].update(v)
            else:
                model_overrides[k] = v
        else:
            model_overrides.update({k:v})

    model_overrides['dataset_reader'].pop('adversarial_examples_path', None)
    sys.argv.extend(['-o', str(model_overrides).replace("True", "'True'").replace("False", "'False'").replace("None", "'None'")])


def launch_baseline_test():
    ix = sys.argv.index('--overrides_file')
    sys.argv.pop(ix)                        # Pop flag
    overrides_file = sys.argv.pop(ix)       # And pop path

    file_overrides = json.loads(_jsonnet.evaluate_file(overrides_file))
    model_overrides = json.load(open(sys.argv[2].replace('model.tar.gz', 'config.json'), 'r'))
    for k,v in file_overrides.items():
        if isinstance(v, dict):
            if k in model_overrides:
                model_overrides[k].update(v)
            else:
                model_overrides[k] = v
        else:
            model_overrides.update({k:v})

    model_overrides['dataset_reader'].pop('adversarial_examples_path', None)
    sys.argv.extend(['-o', str(model_overrides).replace("True", "'True'").replace("False", "'False'").replace("None", "'None'")])


def launch_adversarial_random_benchmark(args):
    ix = sys.argv.index('--overrides_file')
    sys.argv.pop(ix)                        # Pop flag
    overrides_file = sys.argv.pop(ix)       # And pop path
    file_overrides = json.loads(_jsonnet.evaluate_file(overrides_file))

    from ruletaker.allennlp_models.models.adversarial_benchmark import RandomAdversarialBaseline as Model

    file_overrides['model_class'] = Model
    args.config = file_overrides
    return args


if __name__ == '__main__':
    main()
