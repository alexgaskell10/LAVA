import torch
import numpy as np

from allennlp.models.archival import load_archive #archive_model, CONFIG_NAME, 
from allennlp.data.dataloader import DataLoader
from allennlp.nn import util as nn_util
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer as Optimizer
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular as Scheduler
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import GradientDescentTrainer as Trainer

from ruletaker.allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
# from ruletaker.allennlp_models.models.transformer_binary_qa_model import TransformerBinaryQA as Model
from ruletaker.allennlp_models.train.utils import read_pkl

config = {
    "file_path": "./ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/dev.jsonl",
    "dset_config": {
        'add_NAF': False, #True,
        'true_samples_only': False,
        'concat_q_and_c': True,
        'shortest_proof': 1,
        'longest_proof': 10,
        'topk': 10,
        'pretrained_retriever_model': None, #'bin/runs/pretrain_retriever/rb-base/model.tar.gz',
        'retriever_variant': 'roberta-large',
        'sample': -1,
        'use_context_full': False,
        'scramble_context': False,
        'skip_id_regex': '$none',
        'add_prefix': {'c': 'C: ','q': 'Q: '},
        'syntax': 'rulebase',
        'max_pieces': 384,
        'pretrained_model': 'roberta-base',
    },
    "archive_config": {
        "archive_file": "./ruletaker/runs/depth-5-base", #"./ruletaker/runs/depth-5-base", "./ruletaker/runs/depth-5"
        "cuda_device": 2,
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
        'cuda_device': 2,
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
    },
    "scheduler_config": {
        'decay_factor': 0.38,
        'discriminative_fine_tuning': False,
        'gradual_unfreezing': False,
        'last_epoch': -1,
        'ratio': 32,
        'cut_frac': 0.06,
        'num_steps_per_epoch': 7442,
        'num_epochs': 4,
    }
}

archive = load_archive(**config["archive_config"])
model = archive.model.eval()
vocab = model.vocab
cuda_device = config["archive_config"]["cuda_device"]

checkpointer = Checkpointer(config['trainer_config']['serialization_dir'])
optimizer = Optimizer(model_parameters=[[n,p] for n, p in model.named_parameters() if p.requires_grad], **config["optimizer_config"])
learning_rate_scheduler = Scheduler(optimizer=optimizer, **config["scheduler_config"])

dset = DataReader(**config["dset_config"])
# dataset = dset.read(config["file_path"])
dataset = read_pkl(dset, config["file_path"])

dataset.index_with(vocab)
dataloader = DataLoader(dataset=dataset, **config["dataloader_config"])

batches = iter(dataloader)
batch = next(batches)
batch = nn_util.move_to_device(batch, cuda_device)
outputs = model(**batch)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    learning_rate_scheduler=learning_rate_scheduler,
    checkpointer=checkpointer,
    validation_data_loader=dataloader,
    data_loader=dataloader,         # TODO
    **config['trainer_config']
)
# trainer._validation_loss(0)


##### ^^Main allennlp framework code^^ #####


def check_data():
    mono, non_mono = [], []
    for batch in batches:
        meta = batch['metadata'][0]
        label = batch['label'][0]
        sentences = [s.strip()+'.' for s in meta['context'].split('.')[:-1]]
        nodes = meta['node_label']
        question = meta['question_text']
        proof_sentences = [s for s,n in zip(sentences, nodes) if n==1]
        id = meta['id']

        tgt_sentences = [question] + proof_sentences
        neg_sentences = ['not' in t for t in tgt_sentences]

        # print(any(neg_sentences))
        if any(neg_sentences):
            non_mono.append(id)
        else:
            mono.append(id)

    print(len(mono), len(mono+non_mono), len(mono)/len(mono+non_mono))
        

def is_mono(batch):
    meta = batch['metadata'][0]
    label = batch['label'][0]
    sentences = [s.strip()+'.' for s in meta['context'].split('.')[:-1]]
    nodes = meta['node_label']
    question = meta['question_text']
    proof_sentences = [s for s,n in zip(sentences, nodes) if n==1]
    id = meta['id']

    tgt_sentences = [question] + proof_sentences
    neg_sentences = ['not' in t for t in tgt_sentences]

    return not any(neg_sentences)


def binary_vector(nodes):
    ''' Model each sentence using Bernoulli RV. Ensure proof nodes are present. '''
    nodes = np.array(nodes)
    select_idxs = np.random.choice([0,1], nodes.shape)
    select_idxs += nodes
    select_idxs = np.where(select_idxs > 1, 1, select_idxs)
    return select_idxs


def random_selection(nodes):
    ''' Sample one sentence to discard. Ensure proof nodes are present. '''
    nodes = np.array(nodes)
    proof_idxs = nodes.nonzero()[0]
    while True:
        sample = np.random.choice(len(nodes))
        if sample not in proof_idxs:
            break
    select_idxs = np.ones_like(nodes)
    select_idxs[sample] = 0
    return select_idxs


def probe_model(model, select_func):
    model = model.eval()
    aggr_results = []
    while True:
        batch = next(batches)
        batch = nn_util.move_to_device(batch, cuda_device)
        outputs = model(**batch)
        orig_results = outputs['label_probs'][0]
        label = batch['label'][0]

        if orig_results.argmax() == 0 or label == 0:
            continue

        if not is_mono(batch):
            continue

        meta = batch['metadata'][0]
        label = batch['label'][0]
        sentences = [s.strip()+'.' for s in meta['context'].split('.')[:-1]]
        nodes = meta['node_label']
        question = meta['question_text']
        proof_sentences = [s for s,n in zip(sentences, nodes) if n==1]

        sampled_results = []
        for _ in range(20):
            select_idxs = select_func(nodes)
            new_sentences = [s for s,n in zip(sentences, select_idxs) if n==1]
            assert all([p in new_sentences for p in proof_sentences])

            tmp_batch = dset.encode_batch([(question, ' '.join(new_sentences), label)], vocab)
            tmp_batch = nn_util.move_to_device(tmp_batch, cuda_device)
            with torch.no_grad():
                outputs = model(**tmp_batch)
            
            result = outputs['label_probs'][0]
            sampled_results.append([result, select_idxs, new_sentences])
            
        scores = torch.cat([r[0].unsqueeze(0) for r in sampled_results])[:,0]
        flips = (scores > 0.5).sum()
        flip_rate = flips.float() / scores.size(0)
        if flip_rate > 0:
            print(123)

        aggr_results.append([flip_rate.item(), batch['metadata'][0]['QLen']])
        
        print(len(aggr_results))
        if len(aggr_results) == 25:
            break


if __name__ == '__main__':
    func = binary_vector        # random_selection binary_vector
    probe_model(model, func)

    # check_data()
