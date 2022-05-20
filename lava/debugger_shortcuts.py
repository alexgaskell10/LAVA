# A file storing command line args to help with convenient debugging
import sys, os
from .utils import datetime_now


def shortcut_launch(cmds):
    sys.argv[1:] = cmds[sys.argv.pop(1)]
    if 'tmp' in sys.argv[2] or 'tmp' in sys.argv[4]:
        while os.path.isdir(sys.argv[4]):
            if os.path.isdir(sys.argv[4]):
                os.system(f"rm -rf {sys.argv[4]}")
            if os.path.isdir(sys.argv[4]):
                sys.argv[4] += '.1'


outdir_adv = 'resources/runs/adversarial/'+datetime_now()
outdir_rt = 'resources/runs/ruletaker/'+datetime_now()
outdir_random_bl = 'resources/runs/baselines/random_adversarial/'+datetime_now()

debugger_shortcut_cmds = {
    "ruletaker_train_original": ['ruletaker_train_original', 'resources/config/ruletaker/rulereasoning_config_tmp.jsonnet', '-s', outdir_rt],
    "ruletaker_adv_training": ['ruletaker_adv_training', 'resources/config/ruletaker/ruletaker_adv_retraining.jsonnet', '-s', outdir_rt],
    "adversarial_dataset_generation": ['adversarial_dataset_generation', 'resources/config/attacker/tmp.jsonnet', '-s', outdir_adv],
    "ruletaker_adv_training_test": ['ruletaker_adv_training_test', 
        'resources/runs/ruletaker/2021-12-22_09-38-17_distilroberta-base_retrain/model.tar.gz', 'test', 
        '--output-file', 'resources/runs/ruletaker/2021-12-22_09-38-17_distilroberta-base_retrain/aug_test_results.json', 
        '--overrides_file', 'resources/config/ruletaker/ruletaker_adv_retraining_2022-01-10_08-25-19.jsonnet',
        '--cuda-device', '1',
    ],
    "ruletaker_eval_original": ['ruletaker_eval_original',
        'resources/runs/ruletaker/depth-5/model.tar.gz', 'dev', '--output-file', '_results.json', 
        '-o', "{'trainer': {'cuda_device': 3}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
        '--cuda-device', '3',
    ],
    "ruletaker_test_original": ['ruletaker_test_original',
        'resources/runs/ruletaker/depth-5/model.tar.gz', 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl', 
        '--output-file', '_results.json', 
        '-o', "{'trainer': {'cuda_device': 9}, 'validation_data_loader': {'batch_sampler': {'batch_size': 64, 'type': 'bucket'}}}", 
        '--cuda-device', '9',
    ],
    "baseline_test": ['baseline_test',
        'resources/runs/ruletaker/depth-5/model.tar.gz', 'resources/runs/baselines/textfooler/2021-12-21_18-12-03_reevaled.pkl', 
        '--output-file', 'resources/runs/baselines/textfooler/2021-12-21_18-12-03_reevaled_results_abc.pkl', 
        '--overrides_file', 'resources/config/baselines/transferability/config.jsonnet',
        '--cuda-device', '9',
    ],
    "adversarial_dataset_generation_test": ['adversarial_dataset_generation_test',
        'resources/runs/adversarial/2021-12-12_17-38-38_roberta-large/model.tar.gz', 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl', 
        '--output-file', '_results.json',
        '--overrides_file', 'resources/config/attacker/test_config_2022-01-10_08-25-19.jsonnet',
        '--cuda-device', '9',
    ],
    "transferability": ['transferability',
        'resources/runs/ruletaker/2021-12-12_19-08-47_roberta-base/model.tar.gz', 
        'resources/runs/baselines/hotflip//2021-12-16_11-00-14_reevaled_bu.pkl', 
        '--output-file', 'resources/runs/ruletaker/2021-12-12_19-08-47_roberta-base/transferability_results_rb_lg--rb_base_hotflip.json', 
        '--overrides_file', 'resources/config/transferability/config_2021-12-20_20-39-00.jsonnet',
        '--cuda-device', '8',
    ],
    "adversarial_random_benchmark": ["adversarial_random_benchmark",
        '', 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
        '--output-file', f'{outdir_random_bl}_results.json',
        '--overrides_file', 'resources/config/baselines/adversarial_benchmark/config.jsonnet',
        '--cuda-device', '8',
        '--fresh-init',
    ],
}