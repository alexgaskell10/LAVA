local max_pieces = 512;
local skip_id_regex = "$none";
local ruletaker_archive = "ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/";
local retriever_variant = "roberta-large";      // {spacy, roberta-base, roberta-large}
local pretrained_model = "bin/runs/pretrain_retriever/rb-base/model.tar.gz";
local cuda_device = 1;
local batch_size = 4;
local num_gradient_accumulation_steps = 1;
local topk = 5;
local num_monte_carlo = 16;
local longest_proof = topk;
local shortest_proof = 1;
local model_type = 'variational_inference_base';

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "train.jsonl",
    "validation_data_path": dataset_dir + "dev.jsonl",
    "test_data_path": dataset_dir + "test.jsonl",
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": retriever_variant,
        "pretrained_retriever_model": pretrained_model,
        "topk": topk,
        "longest_proof": longest_proof,
        "shortest_proof": shortest_proof,
        "concat_q_and_c": true,
        "true_samples_only": true
    },
    "retrieval_reasoning_model": {
        "variant": retriever_variant,
        "type": model_type,
        "sentence_embedding_method": "mean",
        "topk": topk,
        "num_monte_carlo": num_monte_carlo
    },
    "trainer": {
        "cuda_device": cuda_device,
        "num_gradient_accumulation_steps": num_gradient_accumulation_steps
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