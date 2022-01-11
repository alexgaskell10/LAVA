local ruletaker_archive = 'resources/runs/ruletaker/2022-01-11_14-05-29_distilroberta-base/model.tar.gz';
local val_num_monte_carlo = 1;

{
    "ruletaker_archive": ruletaker_archive,
    "dataset_reader": {
        "max_instances": 10,
        "shortest_proof": 0,
        "longest_proof": 100,
    },
    "retrieval_reasoning_model": {
        "val_num_monte_carlo": val_num_monte_carlo
    },
}