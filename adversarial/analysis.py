import sys, os, pickle

import pandas as pd
import numpy as np

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


if __name__ == '__main__':
    # df = load_as_df(train_path)
    df = load_as_df(val_path)

    df = compute_features(df, 1, 1)
    # df_train = compute_features(df_train, 1)
    print(df.iloc[:,15:].describe())
