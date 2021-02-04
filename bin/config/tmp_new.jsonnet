local max_pieces = 384;
local skip_id_regex = "$none";
local ruletaker_archive = "ruletaker/runs/depth-5-base/model.tar.gz";
local dataset_dir = "ruletaker/inputs/dataset/tiny-rule-reasoning/depth-3ext-NatLang/";
local retriever_variant = "spacy";      // {spacy}
local cuda_device = -1;

{
    "ruletaker_archive": ruletaker_archive,
    "train_data_path": dataset_dir + "dev.jsonl",
    "validation_data_path": dataset_dir + "dev.jsonl",
    "test_data_path": dataset_dir + "dev.jsonl",
    "dataset_reader": {
        "type": "retriever_reasoning",
        "retriever_variant": retriever_variant
    },
    "retrieval_reasoning_model": {
        "variant": retriever_variant,
        "type": "transformer_binary_qa_retriever",
        "sentence_embedding_method": "mean",
        "topk": 5
    },
    "trainer": {
        "cuda_device": cuda_device
    }
}