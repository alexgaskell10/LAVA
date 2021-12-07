import json, os, sys
import torch
import numpy as np

from allennlp.models.archival import load_archive #archive_model, CONFIG_NAME, 
from allennlp.data.dataloader import DataLoader
from allennlp.nn import util as nn_util
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer as Optimizer
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular as Scheduler
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import GradientDescentTrainer as Trainer

sys.path.extend(['/vol/bitbucket/aeg19/re-re'])
# from ruletaker.allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
from ruletaker.allennlp_models.dataset_readers.records_reader import RecordsReader as DataReader
# from ruletaker.allennlp_models.models.transformer_binary_qa_model import TransformerBinaryQA as Model
from ruletaker.allennlp_models.train.utils import read_pkl

cuda_device = 8
config = {
    "file_path": "./ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/dev.jsonl",
    "dset_config": {
        # 'add_NAF': False, #True,
        # 'true_samples_only': False,
        # 'concat_q_and_c': True,
        # 'shortest_proof': 1,
        # 'longest_proof': 100,
        # 'pretrained_retriever_model': None, #'bin/runs/pretrain_retriever/rb-base/model.tar.gz',
        # 'retriever_variant': 'roberta-large',
        'sample': -1,
        'use_context_full': False,
        'scramble_context': False,
        'skip_id_regex': '$none',
        'add_prefix': {'c': 'C: ','q': 'Q: '},
        'syntax': 'rulebase',
        'max_pieces': 384,
        # 'one_proof': True,
        # 'max_instances': 100,
        'pretrained_model': 'roberta-large'
    },
    "archive_config": {
        "archive_file": "./ruletaker/runs/depth-5-base", #"./ruletaker/runs/depth-5-base", "./ruletaker/runs/depth-5"
        "cuda_device": cuda_device,
        "overrides": ""
    },
    "dataloader_config": {   
        'batches_per_epoch': None,
        'multiprocessing_context': None,
        'worker_init_fn': None,
        'timeout': 0,
        'drop_last': False,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
        'batch_size': 1
    },
    "trainer_config": {
        'opt_level': None,
        'num_gradient_accumulation_steps': 1,
        'world_size': 1,
        'local_rank': 0,
        'distributed': None,
        'epoch_callbacks': None,
        'batch_callbacks': None,
        'save_best_model': True,
        'moving_average': None,
        # 'tensorboard_writer': <allennlp.training.tensorboard_writer.TensorboardWriter object at 0x7f8570b1bef0>,
        'momentum_scheduler': None,
        'grad_clipping': 1,
        'grad_norm': None,
        'cuda_device': cuda_device,
        'serialization_dir': 'bin/runs/pretrained_retriever_ruletaker/roberta/tmp1',
        'num_epochs': 4,
        # 'validation_data_loader': <allennlp.data.dataloader.DataLoader object at 0x7f8570b1b940>,
        'validation_metric': '+EM',
        'patience': None,
        # 'data_loader': <allennlp.data.dataloader.DataLoader object at 0x7f8570b1b780>,
    },
    "optimizer_config": {
        'correct_bias': False,
        'weight_decay': 0.1,
        'eps': 1e-06,
        'betas': (0.9, 0.98),
        'lr': 1e-05,
        'parameter_groups': [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
    }
}


if __name__ == '__main__':
    path = 'bin/runs/adversarial/2021-12-03_08-15-20-keep/val-records_epoch2.pkl'
    archive = load_archive(**config["archive_config"])
    model = archive.model.eval()
    vocab = model.vocab
    cuda_device = config["archive_config"]["cuda_device"]

    # checkpointer = Checkpointer(config['trainer_config']['serialization_dir'])
    optimizer = Optimizer(model_parameters=[[n,p] for n, p in model.named_parameters() if p.requires_grad], **config["optimizer_config"])
    # learning_rate_scheduler = Scheduler(optimizer=optimizer, **config["scheduler_config"])

    dset = DataReader(**config["dset_config"])
    dataset = dset.read(path)
    # dataset = read_pkl(dset, config["file_path"])    

    dataset.index_with(vocab)
    dataloader = DataLoader(dataset=dataset, **config["dataloader_config"])

    batches = iter(dataloader)
    batch = next(batches)
    batch = nn_util.move_to_device(batch, cuda_device)
    outputs = model(**batch)

    trainer = Trainer(
        model=model,
        validation_data_loader=dataloader,
        data_loader=dataloader,         # TODO
        optimizer=optimizer,
        **config['trainer_config']
    )
    outputs = trainer._validation_loss(0)
    print('Done')