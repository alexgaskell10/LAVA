local ruletaker_archive = "bin/runs/ruletaker/depth-5/model.tar.gz";
local dataset_dir = "";
local inference_model = "roberta-base";      # {roberta-base, roberta-large}
local batch_size = 16;
local num_monte_carlo = 1;
local longest_proof = 10;
local shortest_proof = 1;
local model_type = 'adversarial_random_benchmark';
local compute_word_overlap_scores = true;

{
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": inference_model,
        "longest_proof": longest_proof,
        "shortest_proof": shortest_proof,
        "one_proof": false,
        "word_overlap_scores": compute_word_overlap_scores,
        "pretrained_model": "roberta-large",
        "max_instances": 1000,
    },
    "model": {
        "ruletaker_archive": ruletaker_archive,
        "variant": inference_model,
        "type": model_type,
        "num_monte_carlo": num_monte_carlo,
        "word_overlap_scores": compute_word_overlap_scores,
        "benchmark_type": "random",      # word_score, random, none
        "adversarial_perturbations": "sentence_elimination,question_flip,equivalence_substitution",       # sentence_elimination,question_flip,equivalence_substitution
        "max_flips": 3,     # -1, 3
        "max_elims": 3,     # -1, 3
    },
    "data_loader": {
        "batch_sampler": {
            "batch_size": batch_size,
            "type": "basic",
            "sampler": "random",        # random, sequential
            "drop_last": false
        }
    }
}