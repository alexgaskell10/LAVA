import json, os, sys
import torch
import yaml
import pickle as pkl
from datetime import datetime

sys.path.extend(['/vol/bitbucket/aeg19/re-re'])

from adversarial.openattack.sort_openattack_import import OpenAttack as oa
from OpenAttack.attack_assist import filter_words

from ruletaker.allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
from ruletaker.allennlp_models.train.utils import read_pkl
from allennlp.models.archival import load_archive
from allennlp.data.dataloader import DataLoader

from adversarial.openattack.custom_victim import CustomVictim as Victim
from adversarial.openattack.custom_tok import CustomTokenizer as Tokenizer
from adversarial.openattack.custom_attackeval import CustomAttackEval as Eval

from datasets import Dataset


config = {
    "file_path": "./ruletaker/inputs/dataset/rule-reasoning-dataset-V2020.2.4/depth-5/dev.jsonl",
    "dset_config": {
        'add_NAF': False, #True,
        'true_samples_only': False,
        'concat_q_and_c': True,
        'shortest_proof': 1,
        'longest_proof': 100,
        'pretrained_retriever_model': None, #'bin/runs/pretrain_retriever/rb-base/model.tar.gz',
        'retriever_variant': 'roberta-large',
        'sample': -1,
        'use_context_full': False,
        'scramble_context': False,
        'skip_id_regex': '$none',
        'add_prefix': {'c': 'C: ','q': 'Q: '},
        'syntax': 'rulebase',
        'max_pieces': 384,
        'one_proof': True,
        'max_instances': False,
        'pretrained_model': 'roberta-large'
    },
    "archive_config": {
        "archive_file": "./ruletaker/runs/depth-5-base", #"./ruletaker/runs/depth-5-base", "./ruletaker/runs/depth-5"
        "cuda_device": 3,
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
    }
}


def datetime_now():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def load_victim_from_archive(config):
    archive = load_archive(**config["archive_config"])
    model = archive.model.eval()
    vocab = model.vocab
    device = torch.device(config["archive_config"]["cuda_device"])

    data_reader = DataReader(**config["dset_config"])

    victim = Victim(model, data_reader, vocab, device)
    
    return victim, data_reader, vocab


def dataset_mapping(x):
    return {
        "x": {"question_text": x["metadata"]["question_text"], "context": x["metadata"]["context"]},
        "target": 1 - x["label"],
        "y": x["label"],
    }


def load_and_remap_config(config):
    mode = sys.argv[1]
    with open(f'bin/config/baselines/{mode}_tmp.yml', 'r') as f:
        manual_config = yaml.load(f, Loader=yaml.BaseLoader)

    max_n = manual_config.pop('max_instances')
    config['dset_config']['max_instances'] = False if max_n == 'False' else int(max_n)
    config['archive_config']['archive_file'] = manual_config.pop('victim_archive_file')
    config['archive_config']['cuda_device'] = int(manual_config.pop('cuda_device'))
    config['attacker'] = manual_config.pop('attacker')
    config['outdir'] = os.path.join(manual_config.pop('outdir'), config['attacker'])

    assert not manual_config

    return config


def dump_results(results, config, attacker, victim, tokenizer):
    os.makedirs(config['outdir'], exist_ok=True)
    atk = type(attacker).__name__
    vic = type(victim).__name__
    tok = type(tokenizer).__name__
    dt = datetime_now()
    save_string = f"{config['outdir']}/{atk}_{vic}_{tok}_{dt}.pkl"

    print('Writing results to:\t', save_string)
    with open(save_string, 'wb') as f:
        pkl.dump(results, f)


if __name__ == '__main__':
    config = load_and_remap_config(config)

    if config['attacker'] == 'hotflip':
        from adversarial.openattack.custom_hotflipattacker import CustomHotFlipAttacker as Attacker
    elif config['attacker'] == 'textfooler':
        from adversarial.openattack.custom_textfoolerattacker import CustomTextFoolerAttacker as Attacker

    # Don't apply perturbs to proper nouns
    filter_words = ['Anne', 'Bob', 'Charlie', 'Dave', 'Erin', 'Fiona', 'Gary', 'Harry']

    victim, data_reader, vocab = load_victim_from_archive(config)

    # Load dataset initially as allennlp dataset (to be compatible with main codebase)
    # and convert to huggingface dataset to be compatible with OpenAttack
    dataset = read_pkl(data_reader, config["file_path"])
    dataset.index_with(vocab)
    config["dataloader_config"]["batch_size"] = len(dataset)                                # OpenAttack doesn't use batches so pass full dataset in as one batch
    dataloader = DataLoader(dataset=dataset, **config["dataloader_config"])
    dset_dict = list(dataloader)[0]
    hf_dataset = Dataset.from_dict({
        'label': dset_dict['label'].data,
        'metadata': dset_dict['metadata'],
        'token_ids': dset_dict['phrase']['tokens']['token_ids'],
        'mask': dset_dict['phrase']['tokens']['mask'],
        'type_ids': dset_dict['phrase']['tokens']['type_ids'],
    })
    hf_dataset = hf_dataset.map(function=dataset_mapping)

    tok = Tokenizer(data_reader)
    attacker = Attacker(tokenizer=tok, filter_words=filter_words)
    attack_eval = Eval(attacker, victim, tokenizer=tok)
    summary, results = attack_eval.eval(hf_dataset, visualize=True)

    dump_results(results, config, attacker, victim, tok)

    print('Done')





    # import OpenAttack as oa
    # import numpy as np
    # import datasets
    # import nltk
    # from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # # configure access interface of the customized victim model by extending OpenAttack.Classifier.
    # class MyClassifier(oa.Classifier):
    #     def __init__(self):
    #         # nltk.sentiment.vader.SentimentIntensityAnalyzer is a traditional sentiment classification model.
    #         nltk.download('vader_lexicon')
    #         self.model = SentimentIntensityAnalyzer()
        
    #     def get_pred(self, input_):
    #         return self.get_prob(input_).argmax(axis=1)

    #     # access to the classification probability scores with respect input sentences
    #     def get_prob(self, input_):
    #         ret = []
    #         for sent in input_:
    #             # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
    #             res = self.model.polarity_scores(sent)

    #             # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
    #             # Adding 10^−6 is a trick to avoid dividing by zero.
    #             prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)

    #             ret.append(np.array([1 - prob, prob]))
            
    #         # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
    #         return np.array(ret)

    # def dataset_mapping(x):
    #     return {
    #         "x": x["sentence"],
    #         "y": 1 if x["label"] > 0.5 else 0,
    #     }
        
    # # load some examples of SST-2 for evaluation
    # dataset = datasets.load_dataset("sst", split="train[:20]").map(function=dataset_mapping)
    # # choose the costomized classifier as the victim model
    # victim = MyClassifier()
    # # choose PWWS as the attacker and initialize it with default parameters
    # # attacker = oa.attackers.PWWSAttacker()
    # attacker = oa.attackers.TextFoolerAttacker()
    # # prepare for attacking
    # attack_eval = oa.AttackEval(attacker, victim)
    # # launch attacks and print attack results 
    # attack_eval.eval(dataset, visualize=True)