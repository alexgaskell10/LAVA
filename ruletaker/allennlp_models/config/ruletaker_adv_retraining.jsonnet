local train_size = 1000; #119068;
local batch_size = 2;
local num_gradient_accumulation_steps = 8;
local num_epochs = 4;
local learning_rate = 1e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local skip_id_regex = "$none";
local transformer_model = "roberta-large";
local max_pieces = 384;
local transformer_weights_model = "ruletaker/runs/depth-5";
local dataset_dir = "ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/";
local cuda_device = 3;

{
  "data_loader": {
    "batch_sampler": {
      "batch_size": batch_size,
      "type": "basic",
      "sampler": "random",      # random,sequential
      "drop_last": false
    }
  },
  "dataset_reader": {
    "type": "blended_rule_reasoning",
    "sample": -1,
    "add_prefix": {"q": "Q: ", "c": "C: "},
    "skip_id_regex": skip_id_regex,
    "pretrained_model": transformer_model,
    "max_pieces": max_pieces,
    "max_instances": "none", #100,none
    "adversarial_examples_path": "bin/runs/adversarial/2021-12-01_17-36-59-keep/train-records_epoch2.pkl", # "none","bin/runs/adversarial/2021-12-01_17-36-59-keep/train-records_epoch2.pkl"
  },
  "train_data_path": dataset_dir + "train.jsonl",
  "validation_data_path": dataset_dir + "dev.jsonl",
  "test_data_path": dataset_dir + "test.jsonl",
  "evaluate_on_test": true,
  "model": {
    "type": "transformer_binary_qa",
    "num_labels": 2,
    "transformer_weights_model": transformer_weights_model,
    "pretrained_model": transformer_model
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "betas": [0.9, 0.98],
      "weight_decay": weight_decay,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": warmup_ratio,
      "num_steps_per_epoch": std.ceil(train_size / (num_gradient_accumulation_steps * batch_size)),
    },
    "validation_metric": "+EM",
    "checkpointer": {
        "num_serialized_models_to_keep": 1
    },
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
    "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  }
}