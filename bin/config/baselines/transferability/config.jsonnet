local batch_size = 16;
local dataset_reader_type = 'baseline_records_reader';

{
    "train_data_path": "",
    "validation_data_path": "",
    "test_data_path": "",
    "dataset_reader": {
        "type": dataset_reader_type,
        "max_instances": 20,
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