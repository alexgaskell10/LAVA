local batch_size = 16;
local adversarial_examples_path_test = "bin/runs/adversarial/2021-12-03_08-15-20-keep/val-records_epoch2.pkl";
local dataset_dir = "data/rule-reasoning-dataset-V2020.2.4/depth-5/";

{
    "train_data_path": "",
    "validation_data_path": "",
    "test_data_path": adversarial_examples_path_test,
    "dataset_reader": {
        "type": "records_reader",
        "max_instances": 40,
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