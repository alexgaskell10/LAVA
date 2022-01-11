import json, os, sys, _jsonnet
import torch
import yaml
import pickle as pkl
from datetime import datetime

sys.path.extend([os.getcwd()])                       # Hack to fix bug with VS code debugger. Ignore

from lava.openattack.sort_openattack_import import OpenAttack as oa
from lava.openattack.config import config
from OpenAttack.attack_assist import filter_words

from lava.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
from lava.train.utils import read_pkl
from allennlp.models.archival import load_archive
from allennlp.data.dataloader import DataLoader

from lava.openattack.custom_victim import CustomVictim as Victim
from lava.openattack.custom_tok import CustomTokenizer as Tokenizer
from lava.openattack.custom_attackeval import CustomAttackEval as Eval

from datasets import Dataset


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
    overrides_file = sys.argv[1]
    manual_config = json.loads(_jsonnet.evaluate_file(overrides_file))

    max_n = manual_config.pop('max_instances')
    config['dset_config']['max_instances'] = False if max_n == 'False' else int(max_n)
    config['archive_config']['archive_file'] = manual_config.pop('victim_archive_file')
    config['archive_config']['cuda_device'] = int(manual_config.pop('cuda_device'))
    config['attacker'] = manual_config.pop('attacker')
    config['outdir'] = os.path.join(manual_config.pop('outdir'), config['attacker'])
    config['outpath'] = manual_config.pop("outpath")
    config['file_path'] = manual_config.pop("file_path")
    config['pkl_path'] = manual_config.pop('pkl_path', None)

    assert not manual_config, manual_config

    return config


def dump_results(results, config, attacker, victim, tokenizer, save_string=None):
    os.makedirs(config['outdir'], exist_ok=True)

    if save_string is None:
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
        from lava.openattack.custom_hotflipattacker import CustomHotFlipAttacker as Attacker
    elif config['attacker'] == 'textfooler':
        from lava.openattack.custom_textfoolerattacker import CustomTextFoolerAttacker as Attacker

    # Don't apply perturbs to proper nouns
    filter_words = ['Anne', 'Bob', 'Charlie', 'Dave', 'Erin', 'Fiona', 'Gary', 'Harry']
    filter_words += ['The', 'the', '.', 'If']

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

    dump_results(results, config, attacker, victim, tok, config['outpath'])
    print(summary)

    print('Done')
