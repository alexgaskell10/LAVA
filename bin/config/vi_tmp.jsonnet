local max_pieces = 512;
local skip_id_regex = "$none";
local ruletaker_archive = "ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/tiny-rule-reasoning/challenge/";
local retriever_variant = "roberta-base";      // {spacy, roberta-base, roberta-large}
local pretrained_model = "bin/runs/pretrain_retriever/rb-base/model.tar.gz";
local cuda_device = 0;
local batch_size = 8;
local num_gradient_accumulation_steps = 1;
local topk = 1;
local longest_proof = topk;
local shortest_proof = 1;
local model_type = 'variational_inference_base';

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "train__.jsonl",
    "validation_data_path": dataset_dir + "test.jsonl",
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
        "topk": topk
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