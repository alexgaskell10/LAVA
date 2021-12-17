local ruletaker_archive = "bin/runs/ruletaker/depth-5/model.tar.gz";
local dataset_dir = "data/rule-reasoning-dataset-V2020.2.4/depth-5/";
local inference_model = "roberta-base";      # {roberta-base, roberta-large}
local cuda_device = 9;
local batch_size = 8;
local num_gradient_accumulation_steps = 1;
local num_monte_carlo = 8;
local longest_proof = 10;
local shortest_proof = 1;
local lr = 5e-6;
local model_type = 'adversarial_base';
local compute_word_overlap_scores = true;
local epochs = 2;
local adversarial_perturbations = "sentence_elimination,question_flip,equivalence_substitution";

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "train.jsonl",
    "validation_data_path": dataset_dir + "dev.jsonl",
    "test_data_path": dataset_dir + "test.jsonl",
    "lr": lr,
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": inference_model,
        "longest_proof": longest_proof,
        "shortest_proof": shortest_proof,
        "one_proof": false,
        "word_overlap_scores": compute_word_overlap_scores,
        "max_instances": -1,
    },
    "retrieval_reasoning_model": {
        "variant": inference_model,
        "type": model_type,
        "num_monte_carlo": num_monte_carlo,
        "word_overlap_scores": compute_word_overlap_scores,
        "benchmark_type": "none",
        "bernoulli_node_prediction_level": "node-level",
        "adversarial_perturbations": adversarial_perturbations,
        "max_flips": 3,
        "max_elims": 3,
    },
    "trainer": {
        "cuda_device": cuda_device,
        "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
        "type": "adversarial_trainer",
        "save_best_model": true,
        "num_epochs": epochs,
        "patience": 2,
        "learning_rate_scheduler": {
            'cut_frac': 0.06,
            'type': 'slanted_triangular',
        }
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