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


def _all(i, seq):
    return all(x == i for x in seq)


def extract_sub(seq_x, seq_y):
    ''' Here:
        - [' ','-'],['a','b'] -> {'a b': 'a'}
        - [' ','-','+'],['a','b','c'] -> {'b': 'c'}
        - [' ','+'], ['a','b'] -> {'a': 'a b'}
    '''
    if seq_x[0] == ' ' and _all('+', seq_x[1:]):
        sub = [seq_y[0], ' '.join(seq_y)]
    elif seq_x[0] == ' ' and _all('-', seq_x[1:]):
        sub = [' '.join(seq_y), seq_y[0]]
    else:
        n_plus = seq_x.count('+')
        sub = [' '.join(seq_y[1:-n_plus]), ' '.join(seq_y[-n_plus:])]
    return sub


def extract_textfooler(orig, adv):
    ''' Extract the substutition required to align the strings.
        - [a,b,c] || [a,d,b,c] -> [a] : [a,d]
        - [a,b,b,c] || [a,d,d,c] -> [b,b] : [d,d]
        - [a,b,c] || [a,c] -> [a,b] : [a]
    '''
    diffs = list(difflib.ndiff(orig, adv))
    xs = [x[0] for x in diffs]
    ys = [x[2:] for x in diffs]
    if '?' in xs:
        return {}
    subs = {}
    prev_x = ' '
    seq_x, seq_y = [], []
    sent_num = 0
    for i in range(len(diffs)):
        if i > 0 and ys[i-1] == '.':
            sent_num += 1

        if xs[i] == prev_x == ' ':
            continue

        if xs[i] == prev_x != ' ':
            seq_x.append(xs[i])
            seq_y.append(ys[i])
            continue

        if xs[i] != ' ' == prev_x:
            if i == 0:
                seq_x.append(xs[i])
                seq_y.append(ys[i])
            else:
                seq_x.extend(xs[i-1: i+1])
                seq_y.extend(ys[i-1: i+1])
            prev_x = xs[i]
            continue

        if xs[i] != ' ' != prev_x:
            seq_x.append(xs[i])
            seq_y.append(ys[i])
            continue

        if xs[i] == ' ' != prev_x:
            sub = extract_sub(seq_x, seq_y)
            subs.update({sub[0]: [sub[1], sent_num]})
            seq_x, seq_y = [], []
            prev_x = xs[i]
            continue

    return subs


def process_meta(config, adversarial_data):
    data_reader = DataReader(**config["dset_config"])

    proc_data = []
    for n, data in enumerate(adversarial_data):
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
            meta['meta_record'][meta_type][meta_id] = new_meta
        elif config['attacker'] == 'textfooler':
            subs = extract_textfooler(orig_, adv_)
            if len(subs) == 0:
                meta['skip'] = True
                proc_data.append(meta)
                continue
            try:
                for k,v in subs.items():
                    sent_ix = v[1]
                    sub = [k, v[0]]
                    orig_sent = orig.split('.')[sent_ix].strip() + '.'
                    meta_id = [
                        k for k,v in {**meta_record['rules'], **meta_record['triples']}.items()
                        if v is not None and v['text'].replace(' ','') == orig_sent.replace(' ','')][0]
                    meta_type = ''.join([x for x in meta_id if not x.isnumeric()]) + 's'
                    new_meta = meta_record[meta_type][meta_id]
                    new_meta['representation'] = new_meta['representation'].replace(*sub)
                    new_meta['text'] = new_meta['text'].replace(*sub)
                    meta['meta_record'][meta_type][meta_id] = new_meta
                    continue
            except IndexError:
                meta['skip'] = True
                proc_data.append(meta)
                continue

        meta['meta_record']['questions'] = {qid: meta['meta_record']['questions'][qid]}
        meta['meta_record']['triples'] = add_mask(meta['meta_record']['triples'])
        meta['meta_record']['rules'] = add_mask(meta['meta_record']['rules'])
        meta['skip'] = False
        proc_data.append(meta)
        continue
    
    return proc_data


def add_mask(rules_or_triples):
    return {k:{**v, 'mask':0} for k,v in rules_or_triples.items() if v is not None}


if __name__ == '__main__':
    # path = 'bin/runs/baselines/hotflip/CustomHotFlipAttacker_CustomVictim_CustomTokenizer_2021-12-06_13-36-26.pkl'
    path = 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05.pkl'
    config['attacker'] = 'textfooler' if 'textfooler' in path.lower() else 'hotflip'
    with open(path, 'rb') as f:
        adv_data = pkl.load(f)[:500]

    proc_data = process_meta(config, adv_data)
    engine_labels = [
        call_theorem_prover_from_lst([d['meta_record']])[0] 
        if d['result'] is not None and not d['skip']
        else None 
        for d in proc_data]

    # Compute modified results: correct if engine_label == (adv result == 
    adv_results = [d['metrics']['Succeed'] for d in adv_data]
    orig_labels = [p['label'] for p in proc_data]
    modified_correct = [a and o==e for a,o,e in zip(adv_results, orig_labels, engine_labels) if e is not None]
    modified = [e for e in engine_labels if e is not None]

    unadj_flip_rate = adv_results.count(True) / len(adv_results)
    adj_flip_rate = modified_correct.count(True) / len(modified)

    out_str = f'Unadjusted flip rate: {unadj_flip_rate}\nAdjusted flip rate: {adj_flip_rate}\nNum samples before adjustment: {len(adv_data)}\nNum samples post adjustment: {len(modified)}'
    print(out_str)
    with open(path.replace('.pkl', '_modresults.txt'), 'w') as f:
        f.write(out_str)
