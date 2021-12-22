local ruletaker_archive = 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/model.tar.gz';
local val_num_monte_carlo = 1;

{
    "ruletaker_archive": ruletaker_archive,
    "dataset_reader": {
        "max_instances": -1,
        "shortest_proof": 0,
        "longest_proof": 100,
    },
    "retrieval_reasoning_model": {
        "val_num_monte_carlo": val_num_monte_carlo
    },
}