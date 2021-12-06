import json, os, sys
import torch
import yaml
import pickle as pkl
from datetime import datetime
import difflib
# from sklearn.metrics import accuracy_scores

sys.path.extend(['/vol/bitbucket/aeg19/re-re'])

from ruletaker.allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
from ruletaker.allennlp_models.models.ruletaker.theory_label_generator import call_theorem_prover_from_lst

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

def extract_sent(tok_txt, ix):
    eos = ix + tok_txt[ix:].index('.') + 1
    try:
        bos = next(i for i in reversed(range(len(tok_txt[:ix]))) if tok_txt[i] == '.') + 1
    except StopIteration:
        bos = 0
    return ' '.join(tok_txt[bos:eos]).replace(' .','.')


def process_meta(config, adversarial_data):
    data_reader = DataReader(**config["dset_config"])

    proc_data = []
    for data in adversarial_data:
        adv = data['result']
        meta = data['data']['metadata']
        meta['result'] = adv
        if adv is None:
            proc_data.append(meta)
            continue
        orig = meta['context']
        qid = 'Q' + meta['id'].split('-')[-1]
        meta_record = meta['meta_record']
        adv_ = data_reader.tok(adv)
        orig_ = data_reader.tok(orig)
        if config['attacker'] == 'hotflip':
            ix = data['counter']
            sub = [orig_[ix], adv_[ix]]
            orig_sent = extract_sent(orig_, ix)
            meta_id = [
                k for k,v in {**meta_record['rules'], **meta_record['triples']}.items()
                if v is not None and v['text'].replace(' ','') == orig_sent.replace(' ','')][0]
            meta_type = ''.join([x for x in meta_id if not x.isnumeric()]) + 's'
            new_meta = meta_record[meta_type][meta_id]
            new_meta['representation'] = new_meta['representation'].replace(*sub)
            new_meta['text'] = new_meta['text'].replace(*sub)
        elif config['attacker'] == 'textfooler':
            pass
            # difflib.ndiff(orig_, adv_)
            # orig_subs, current_sub = [], []       # First split into sublists based on sentences. Then use difflib on sentence pairs to get the sub. Then apply to logic prog
            # for i in range(len(orig_)):

        meta['meta_record'][meta_type][meta_id] = new_meta
        meta['meta_record']['questions'] = {qid: meta['meta_record']['questions'][qid]}
        meta['meta_record']['triples'] = add_mask(meta['meta_record']['triples'])
        meta['meta_record']['rules'] = add_mask(meta['meta_record']['rules'])
        proc_data.append(meta)
        continue
    
    return proc_data


def add_mask(rules_or_triples):
    return {k:{**v, 'mask':0} for k,v in rules_or_triples.items() if v is not None}

if __name__ == '__main__':
    # path = 'bin/runs/baselines/hotflip/CustomHotFlipAttacker_CustomVictim_CustomTokenizer_2021-12-05_17-56-08.pkl'
    path = 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-05_21-49-19.pkl'
    config['attacker'] = 'textfooler' if 'textfooler' in path.lower() else 'hotflip'
    with open(path, 'rb') as f:
        adv_data = pkl.load(f)

    proc_data = process_meta(config, adv_data)
    engine_labels = [
        call_theorem_prover_from_lst([d['meta_record']])[0] 
        if d['result'] is not None
        else False 
        for d in proc_data]

    # Compute modified results: correct if engine_label == (adv result == 
    adv_results = [d['metrics']['Succeed'] for d in adv_data]
    orig_labels = [p['label'] for p in proc_data]
    modified_correct = [a and o==e for a,o,e in zip(adv_results, orig_labels, engine_labels)]

    unadj_flip_rate = adv_results.count(True) / len(adv_results)
    adj_flip_rate = modified_correct.count(True) / len(modified_correct)

    out_str = f'Unadjusted flip rate: {unadj_flip_rate}\nAdjusted flip rate: {adj_flip_rate}\nNum samples: {len(adv_data)}'
    print(out_str)
    with open(path.replace('.pkl', '_modresults.txt'), 'w') as f:
        f.write(out_str)
