import sys, os, pickle

import pandas as pd
import numpy as np

from math import ceil, floor

# train_path = 'bin/runs/adversarial/2021-12-01_17-36-59-keep/train-records_epoch1.pkl'
val_path = 'bin/runs/adversarial/2021-12-03_08-15-20-keep/val-records_epoch-1.pkl'

def load_as_df(path):
    records = pickle.load(open(path, 'rb'))
    return pd.DataFrame.from_records(records)
    

def compute_features(df, skip_start, skip_end):
    ''' Compute features of interest and add to dataframe.
        -skip_start: # context sentences to skip (i.e. should be 1 when 
        the question is appended to the beginning of the context)
        -skip_end: # context sentences to skip (i.e. should be 1 when 
        the question is appended to the beginning of the context)
    '''
    # Cleaning
    df['sampled_question'] = df['question']
    df['orig_question'] = df['orig_sentences'].apply(lambda x: x[0])
    df['orig_sentences'] = df['orig_sentences'].apply(lambda x: x[skip_start : -skip_end])
    df['sampled_sentences'] = df['sampled_sentences'].apply(lambda x: x[skip_start : -skip_end])
    df['orig_proof_len'] = df['proof_sentences'].apply(len)
    # Compute basic features
    df['n_orig_sentences'] = df['orig_sentences'].apply(len)
    df['n_sampled_sentences'] = df['sampled_sentences'].apply(len)
    df['orig_tokens'] = df['orig_sentences'].apply(lambda x: ' '.join(x)).str.split(' ').apply(len)
    df['sampled_tokens'] = df['sampled_sentences'].apply(lambda x: ' '.join(x)).str.split(' ').apply(len)
    # Compute context composition features
    df['n_orig_rules'] =  df['orig_sentences'].apply(lambda sents: len([s for s in sents if any(i in s for i in ['All','If','are'])]))
    df['n_orig_facts'] = df['n_orig_sentences'] - df['n_orig_rules']
    df['n_sampled_rules'] = df['sampled_sentences'].apply(lambda sents: len([s for s in sents if any(i in s for i in ['All','If','are'])]))
    df['n_sampled_facts'] = df['n_sampled_sentences'] - df['n_sampled_rules']
    # Compute precision, recall, f1
    df['retained_sentences'] = df[['orig_sentences', 'sampled_sentences']].apply(lambda sents: set(sents[0]).intersection(set(sents[1])), axis=1)
    df['sampled_precision'] = df['retained_sentences'].apply(len) / df['n_orig_sentences']
    df['sampled_recall'] = df['retained_sentences'].apply(len) / df['n_sampled_sentences']
    df['sampled_f1'] = 2 * df['sampled_precision'] * df['sampled_recall'] / (df['sampled_precision'] + df['sampled_recall'])
    # Process latent variables
    df['n_sentelim'] = df['z_sent'].apply(lambda z: len([x for x in z if x!=-1]))
    df['n_eqivsubt'] = df['z_eqiv'].apply(lambda z: len([x for x in z if x!=-1]))
    df['n_quesflip'] = df['z_ques'].apply(lambda x: x == [0]).astype(int)
    return df


# def get_examples(df):


def wrap(lst, max_chars):
    new_lst = []
    insert = '\t'
    for sent in lst:
        n_wraps = floor(len(sent) / max_chars)
        for i in range(n_wraps+1):
            new_lst.append(int(i>0)*insert+sent[i*(max_chars+len(insert)):(i+1)*(max_chars+len(insert))])
            # sent[:i*(max_chars+len(insert))] + insert + sent[i*(max_chars+len(insert)):]
        new_lst.append(sent)
    return new_lst


if __name__ == '__main__':
    # df = load_as_df(train_path)
    df = load_as_df(val_path)

    df = compute_features(df, 1, 1)
    # df_train = compute_features(df_train, 1)
    print(df.iloc[:,17:].describe())

    df_pos = df[df.qa_fooled]

    open('table.txt', 'w')
    from tabulate import tabulate
    for i in range(5):
        to_print = []
        row = df_pos[df_pos.orig_proof_depth == i].iloc[0]
        orig_sents = [x.strip() for x in row.orig_sentences]
        sampled_sents = [x.strip() for x in row.sampled_sentences]
        to_print.append([f'Original Example {i+1}', None])
        to_print.append([None, 'Label', bool(row.label)])
        to_print.append([None, 'Question', row.orig_question])
        to_print.append([None, 'Context', '\n'.join(sorted(orig_sents, key=len))])
        to_print.append([f'Adversarial Example {i+1}', None])
        to_print.append([None, 'Label', bool(1-row.mod_label)])
        to_print.append([None, 'Question', row.sampled_question])
        to_print.append([None, 'Context', '\n'.join(sorted(sampled_sents, key=len))])
        print(tabulate(to_print))

        # with open('table.txt', 'a') as f:
        #     f.write(tabulate(to_print))

    out_df = pd.DataFrame(to_print)
    out_df.to_csv('table.csv')