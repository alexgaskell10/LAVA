local max_pieces = 256;
local skip_id_regex = "$none";
local ruletaker_archive = "ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/";
local retriever_variant = "roberta-base";      // {spacy, roberta-base, roberta-large}
local pretrained_model = "bin/runs/pretrain_retriever/rb-base/model.tar.gz";
local cuda_device = 0;
local batch_size = 4;
local num_gradient_accumulation_steps = 1;
local topk = 5;
local model_type = 'gumbel_softmax_unified';

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
        "concat_q_and_c": false
    },
    "retrieval_reasoning_model": {
        "variant": retriever_variant,
        "type": model_type,
        "sentence_embedding_method": "mean",
        "topk": topk
    },
    "trainer": {
        "cuda_device": cuda_device,
        "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
        "optimizer": {
            "lr": 1e-4
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