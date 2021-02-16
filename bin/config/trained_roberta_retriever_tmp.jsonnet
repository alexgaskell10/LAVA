local max_pieces = 384;
local skip_id_regex = "$none";
local ruletaker_archive = "ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/tiny-rule-reasoning/depth-3ext-NatLang/"; // "ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/";
local retriever_variant = "roberta-base";      // {spacy, roberta-base, roberta-large}
local pretrained_model = "bin/runs/pretrain_retriever/test/model.tar.gz";
local cuda_device = 0;
local batch_size = 16;
local num_gradient_accumulation_steps = 1;
local topk = 5;

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "dev.jsonl", // "train.jsonl",
    "validation_data_path": dataset_dir + "dev.jsonl",
    "test_data_path": dataset_dir + "dev.jsonl", //"test.jsonl",
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": retriever_variant,
        "pretrained_retriever_model": pretrained_model
    },
    "retrieval_reasoning_model": {
        "variant": retriever_variant,
        "type": "transformer_binary_qa_retriever",
        "sentence_embedding_method": "mean",
        "topk": topk,
        "pretrained_retriever_model": pretrained_model
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