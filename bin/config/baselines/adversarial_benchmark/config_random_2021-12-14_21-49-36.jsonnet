local benchmark_variation = "random";       # word_score, random, none
local victim_archive_file = "bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/"; #"bin/runs/ruletaker/depth-5/model.tar.gz";
local inference_model = "roberta-base";

{
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": inference_model,
        "longest_proof": 10,
        "shortest_proof": 1,
        "one_proof": false,
        "word_overlap_scores": true,
        "pretrained_model": "roberta-large",
        "max_instances": -1,
        "add_prefix": {"q": "Q: ", "c": "C: "},
        "skip_id_regex": "$none",
    },
    "model": {
        "ruletaker_archive": victim_archive_file,
        "variant": inference_model,
        "type": 'adversarial_random_benchmark',
        "num_monte_carlo": 1,
        "word_overlap_scores": true,
        "benchmark_type": benchmark_variation,
        "adversarial_perturbations": "sentence_elimination,question_flip,equivalence_substitution",
        "max_flips": 3,
        "max_elims": 3,
    },
    "data_loader": {
        "batch_sampler": {
            "batch_size": 8,
            "type": "basic",
            "sampler": "sequential",
            "drop_last": false
        }
    }
}