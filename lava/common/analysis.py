import sys, os, pickle

import pandas as pd
import numpy as np

from math import ceil, floor

# train_path = 'resources/runs/adversarial/2021-12-01_17-36-59-keep/train-records_epoch1.pkl'
val_path = 'resources/runs/adversarial/2021-12-03_08-15-20-keep/val-records_epoch-1.pkl'

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
    df['n_orig_tokens'] = df['orig_sentences'].apply(lambda sent: sum(len(x.strip().split()) for x in sent))
    df['n_sampled_tokens'] = df['sampled_sentences'].apply(lambda sent: sum(len(x.strip().split()) for x in sent))
    df['orig_tokens'] = df['orig_sentences'].apply(lambda x: ' '.join(x)).str.split(' ').apply(len)
    df['sampled_tokens'] = df['sampled_sentences'].apply(lambda x: ' '.join(x)).str.split(' ').apply(len)
    # Compute context composition features
    df['n_orig_rules'] =  df['orig_sentences'].apply(lambda sents: len([s for s in sents if any(i in s for i in ['All','If','are'])]))
    df['n_orig_facts'] = df['n_orig_sentences'] - df['n_orig_rules']
    df['orig_facts_ixs'] =  df['orig_sentences'].apply(lambda sents: [n for n,s in enumerate(sents) if not any(i in s for i in ['All','If','are'])])
    df['orig_rules_ixs'] =  df['orig_sentences'].apply(lambda sents: [n for n,s in enumerate(sents) if any(i in s for i in ['All','If','are'])])
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

    out_dfs = []
    for i in range(6):
        row = df_pos[df_pos.orig_proof_depth == i].iloc[0]

        orig_sents = ['Context'] + sorted([x.strip()+'.' for x in row.orig_sentences], key=len)
        meta = ['Ques. number', str(int(i))+'-a', '', 'Qid', row.id+'-a', '', 'Claim', row.orig_question, '', 'Answer?', '<ANSWER HERE T/F>']
        meta += [''] * (len(orig_sents) - len(meta))
        orig_rows = pd.DataFrame({0:['-'*15]+meta+['']*2, 1:['-'*15]+orig_sents+['']*2})

        sampled_sents = ['Context'] + sorted([x.strip()+'.' for x in row.sampled_sentences], key=len)
        meta = ['Ques. number', str(int(i))+'-b', '', 'Qid', row.id+'-b', '', 'Claim', row.sampled_question, '', 'Answer?', '<ANSWER HERE T/F>']
        meta += [''] * (len(sampled_sents) - len(meta))
        sampled_rows = pd.DataFrame({0:meta+['']*2, 1:sampled_sents+['']*2})

        qual_meta = ['Which was more challenging?', f"{str(int(i))+'-a'} or {str(int(i))+'-b'}", '<ANSWER HERE>']
        qual_rows = pd.DataFrame({0:qual_meta+['-'*15]+['']*10, 1:['']*len(qual_meta)+['-'*15]+['']*10})

        rows = pd.concat([orig_rows, sampled_rows, qual_rows], axis=0)
        out_dfs.append(rows)

    out_df = pd.concat(out_dfs)
    out_df.to_csv('table.csv', header=False, index=None)

        # orig_rows = []
        # orig_rows.append(['Qnum', 'Qid', 'Question', 'Context', 'Label?'])
        # orig_sents = sorted([x.strip() for x in row.orig_sentences], key=len)
        # orig_rows.append([str(int(i)), row.id, row.orig_question, orig_sents[0], ''])
        # for sent in orig_sents[1:]:
        #     orig_rows.append(['', '', '', sent, ''])
        # out_dfs.append(pd.DataFrame(orig_rows[1:], columns=orig_rows[0]))

        # with open('table.txt', 'a') as f:
        #     f.write(tabulate(to_print))

    # out_df = pd.DataFrame(to_print)
    # out_df.to_csv('table.csv')

    # open('table.txt', 'w')
    # from tabulate import tabulate
    # for i in range(5):
    #     to_print = []
    #     row = df_pos[df_pos.orig_proof_depth == i].iloc[0]
    #     orig_sents = [x.strip() for x in row.orig_sentences]
    #     sampled_sents = [x.strip() for x in row.sampled_sentences]
    #     to_print.append([f'Original Example {i+1}', None])
    #     to_print.append([None, 'Label', bool(row.label)])
    #     to_print.append([None, 'Question', row.orig_question])
    #     to_print.append([None, 'Context', '\n'.join(sorted(orig_sents, key=len))])
    #     to_print.append([f'Adversarial Example {i+1}', None])
    #     to_print.append([None, 'Label', bool(1-row.mod_label)])
    #     to_print.append([None, 'Question', row.sampled_question])
    #     to_print.append([None, 'Context', '\n'.join(sorted(sampled_sents, key=len))])
    #     print(tabulate(to_print))

    #     # with open('table.txt', 'a') as f:
    #     #     f.write(tabulate(to_print))

    # out_df = pd.DataFrame(to_print)
    # out_df.to_csv('table.csv')