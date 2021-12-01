local max_pieces = 512;
local ruletaker_archive = "ruletaker/runs/depth-5/model.tar.gz"; #"ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/"; #"ruletaker/inputs/dataset/tiny-rule-reasoning/challenge/";
local inference_model = "roberta-base";      # {roberta-base, roberta-large}
local pretrained_model = "bin/runs/pretrain_retriever/rb-base/model.tar.gz";
local cuda_device = 5;
local batch_size = 8;
local num_gradient_accumulation_steps = 1;
local num_monte_carlo = 8;
local longest_proof = 10;
local shortest_proof = 1;
local lr = 5e-6;
local model_type = 'adversarial_base';
local add_naf = false;
local compute_word_overlap_scores = true;
local epochs = 3;

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "train.jsonl",
    "validation_data_path": dataset_dir + "dev.jsonl",
    "test_data_path": dataset_dir + "test.jsonl",
    "lr": lr,
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": inference_model,
        "pretrained_retriever_model": pretrained_model,
        "longest_proof": longest_proof,
        "shortest_proof": shortest_proof,
        "concat_q_and_c": true,
        "true_samples_only": false,
        "add_NAF": add_naf,
        "one_proof": true,
        "word_overlap_scores": compute_word_overlap_scores,
        "max_instances": false,     # 10, 100, false
    },
    "retrieval_reasoning_model": {
        "variant": inference_model,
        "type": model_type,
        "num_monte_carlo": num_monte_carlo,
        "add_NAF": add_naf,
        "word_overlap_scores": compute_word_overlap_scores,
        "benchmark_type": "none",      # word_score, random, none
        "bernoulli_node_prediction_level": "node-level",       # sequence-level, node-level
        "adversarial_perturbations": "equivalence_substitution"       # sentence_elimination,question_flip,equivalence_substitution
    },
    "trainer": {
        "cuda_device": cuda_device,
        "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
        "type": "adversarial_trainer",
        "save_best_model": true,
        "num_epochs": epochs,
        "learning_rate_scheduler": {
            'cut_frac': 0.06,
            'type': 'slanted_triangular',
        }
    },
    "data_loader": {
        "batch_sampler": {
            "batch_size": batch_size,
            "type": "basic",
            "sampler": "sequential",
            "drop_last": false
        }
    }
}