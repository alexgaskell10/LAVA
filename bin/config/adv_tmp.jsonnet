local max_pieces = 512;
local skip_id_regex = "$none";
local ruletaker_archive = "ruletaker/runs/depth-5/model.tar.gz"; #"ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/"; #"ruletaker/inputs/dataset/tiny-rule-reasoning/challenge/";
local retriever_variant = "roberta-large";      # {roberta-base, roberta-large}
local pretrained_model = "bin/runs/pretrain_retriever/rb-base/model.tar.gz";
local cuda_device = 1;
local batch_size = 8;
local num_gradient_accumulation_steps = 1;
local topk = 10;
local num_monte_carlo = 8;
local longest_proof = topk;
local shortest_proof = 1;
local lr = 5e-6;
local model_type = 'adversarial_base';
local add_naf = false;
local compute_word_overlap_scores = true;

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "train.jsonl",
    "validation_data_path": dataset_dir + "dev.jsonl",
    "test_data_path": dataset_dir + "test.jsonl",
    "lr": lr,
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": retriever_variant,
        "pretrained_retriever_model": pretrained_model,
        "topk": topk,
        "longest_proof": longest_proof,
        "shortest_proof": shortest_proof,
        "concat_q_and_c": true,
        "true_samples_only": true,
        "add_NAF": add_naf,
        "one_proof": true,
        "word_overlap_scores": compute_word_overlap_scores,
        "max_instances": false,
    },
    "retrieval_reasoning_model": {
        "variant": retriever_variant,
        "type": model_type,
        "sentence_embedding_method": "mean",
        "topk": topk,
        "num_monte_carlo": num_monte_carlo,
        "do_mask_z": true,
        "additional_qa_training": false,
        "objective": "NVIL",       # VIMCO; NVIL
        "sampling_method": "multinomial",        # multinomial; gumbel_softmax; argmax
        "baseline_type": "Prob-NMN",
        "infr_supervision": false,
        "add_NAF": add_naf,
        "threshold_sampling": true,
        "word_overlap_scores": compute_word_overlap_scores,
        "benchmark_type": "random",      # word_score, random
        "bernoulli_node_prediction_level": "node-level",       # sequence-level, node-level
    },
    "trainer": {
        "cuda_device": cuda_device,
        "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
        "type": "adversarial_trainer",
        "save_best_model": false,
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