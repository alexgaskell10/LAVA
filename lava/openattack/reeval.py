import json, os, sys
import torch
import yaml
import pickle as pkl
from datetime import datetime
import difflib

sys.path.extend([os.getcwd()])                       # Hack to fix bug with VS code debugger. Ignore     # TODO

from lava.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader as DataReader
from lava.models.solver.theory_label_generator import call_theorem_prover_from_lst
from lava.openattack.config import config
from lava.openattack.openattack import load_and_remap_config


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
            meta['subs'] = 0
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
            meta['subs'] = 1
        elif config['attacker'] == 'textfooler':
            subs = extract_textfooler(orig_, adv_)
            if len(subs) == 0:
                meta['skip'] = True
                meta['subs'] = 0
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
                meta['subs'] = len(subs)
            except IndexError:
                meta['skip'] = True
                meta['subs'] = 0
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
    config = load_and_remap_config(config)
    config['attacker'] = 'textfooler' if 'textfooler' in config['pkl_path'].lower() else 'hotflip'
    with open(config['pkl_path'], 'rb') as f:
        adv_data = pkl.load(f)

    proc_data = process_meta(config, adv_data)
    engine_labels = [
        call_theorem_prover_from_lst([d['meta_record']])[0] 
        if d['result'] is not None and not d['skip']
        else None 
        for d in proc_data]

    # Compute modified results
    adv_results = [d['metrics']['Succeed'] for d in adv_data]
    orig_labels = [p['label'] for p in proc_data]
    modified_correct = [a and o==e for a,o,e in zip(adv_results, orig_labels, engine_labels) if e is not None]
    modified = [e for e in engine_labels if e is not None]

    unadj_flip_rate = adv_results.count(True) / len(adv_results)
    adj_flip_rate = modified_correct.count(True) / len(modified)

    out_str = f'Unadjusted flip rate: {unadj_flip_rate}\nAdjusted flip rate: {adj_flip_rate}\nNum samples before adjustment: {len(adv_data)}\nNum samples post adjustment: {len(modified)}'
    print(out_str)
    with open(config['outpath'], 'w') as f:
        f.write(out_str)

    # Add labels to dict for later analysis
    proc_data = [{**d, 'adv_result':a, 'mod_label':e} for d,a,e, in zip(proc_data, adv_results, engine_labels)]
    with open(config['pkl_path'].replace('.pkl', '_reevaled.pkl'), 'wb') as f:
        pkl.dump(proc_data, f)
    