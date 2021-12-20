local batch_size = 16;
local adversarial_examples_path_test = 'bin/runs/baselines/hotflip/2021-12-16_11-00-14_reevaled.pkl';
local dataset_dir = "data/rule-reasoning-dataset-V2020.2.4/depth-5/";

{
    "train_data_path": "",
    "validation_data_path": "",
    "test_data_path": dataset_dir + "test.jsonl",
    "dataset_reader": {
        "type": "baseline_rule_reasoning",
        "max_instances": "none",
        "adversarial_examples_path_train": "none",
        "adversarial_examples_path_val": "none",
        "adversarial_examples_path_test": adversarial_examples_path_test,
        "adversarial_examples_path": '',
    },
    "data_loader": {
        "batch_sampler": {
            "batch_size": batch_size,
            "type": "basic",
            "sampler": "random",        # random, sequential
            "drop_last": false
        }
    },
}