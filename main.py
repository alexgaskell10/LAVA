import sys, os
import logging
import _jsonnet, json

from allennlp.common.util import import_module_and_submodules
from allennlp.commands import create_parser
from lava.utils import dict_to_str, nested_args_update
from lava.debugger_shortcuts import debugger_shortcut_cmds, shortcut_launch

import lava

logger = logging.getLogger(__name__)


CMD_MAP = {
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


def main(prog = None):

    if len(sys.argv) == 2:
        # Hacky way to conveniently pre-load a set of commands to use
        # with the vs code debugger
        shortcut_launch(debugger_shortcut_cmds)

    # Remap from program name to main control flow
    prog_name = sys.argv[1]
    sys.argv[1] = CMD_MAP[prog_name]

    # Load args
    parser = create_parser(prog)
    args = parser.parse_args()
    logging.info('Args loaded')

    # Perform task-dependent manual adjustments/overrides of args
    if prog_name in ['adversarial_dataset_generation_test', 'transferability' 
        'ruletaker_adv_training_test', 'baseline_test']:
        args = pre_launch(args, prog_name)
    elif prog_name == 'adversarial_random_benchmark':
        args = launch_adversarial_random_benchmark(args)

    args.func(args)


def pre_launch(args, prog_name):

    # Load manually specified overrides 
    file_overrides = json.loads(_jsonnet.evaluate_file(args.overrides_file))

    # Grab archived model's training params so these can be re-used
    model_overrides_file = args.archive_file.replace('model.tar.gz', 'config.json')
    model_overrides = json.load(open(model_overrides_file, 'r'))
    
    # Override the model's training params with the manually-specified overrides
    model_overrides = nested_args_update(file_overrides, model_overrides)
    
    # Additional arg correction
    if prog_name == 'adversarial_dataset_generation_test':
        model_overrides['model'] = model_overrides.pop('retrieval_reasoning_model')
    elif prog_name in ['transferability' 'ruletaker_adv_training_test', 'baseline_test']:
        model_overrides['dataset_reader'].pop('adversarial_examples_path', None)

    # Reformat args as a string so they can be loaded correctly
    args.overrides = dict_to_str(model_overrides)

    return args


def launch_adversarial_random_benchmark(args):
    # Load manually specified overrides 
    file_overrides = json.loads(_jsonnet.evaluate_file(args.overrides_file))

    # Manually pass in the model class directly as we will be initialising 
    # a fresh version of this model
    from lava.models.adversarial_benchmark import RandomAdversarialBaseline as Model
    file_overrides['model_class'] = Model
    args.config = file_overrides

    return args


if __name__ == '__main__':
    main()
