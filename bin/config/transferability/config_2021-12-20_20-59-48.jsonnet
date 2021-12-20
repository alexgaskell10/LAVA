local batch_size = 16;
local adversarial_examples_path_test = "";
local dataset_dir = "data/rule-reasoning-dataset-V2020.2.4/depth-5/";
local dataset_reader_type = baseline_records_reader;

{
    "train_data_path": "",
    "validation_data_path": "",
    "test_data_path": adversarial_examples_path_test,
    "dataset_reader": {
        "type": dataset_reader_type,
        "max_instances": -1,
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