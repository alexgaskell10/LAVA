import json, os, sys
import pickle as pkl
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from analysis import load_as_df, compute_features

paths = {
    'v_rb-lg': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/test_results.json',
    'v_rb-b': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/test_results.json',
    'v_rb-lg:a_rb-b': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
    'v_rb-b:a_rb-b': 'bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/test_results-records.pkl',
    # 'v_rb-lg:a_hotflip': 'bin/runs/baselines/hotflip/CustomHotFlipAttacker_CustomVictim_CustomTokenizer_2021-12-06_13-36-26_modresults.txt',    # TODO update
    'v_rb-lg:a_textfooler': 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05_reevaled.pkl',   # TODO update
    'v_rb-lg:a_random': 'bin/runs/baselines/random_adversarial/2021-12-14_12-45-49_results-records.pkl',
    'v_rb-lg:a_wordscore': 'bin/runs/baselines/random_adversarial/word_score_2021-12-14_21-49-36_results-records.pkl',
    'v_rb-lg_retrain_orig': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain/test-orig-records_epoch100.pkl',
    'v_rb-lg_retrain_adv': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain/test-adv-records_epoch100.pkl',
    # 'v_rb-lg_retrain_aug': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain/test-aug-records_epoch100.pkl',
    # Transferability
    'trans-v_rb-lg:v_rb-b': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/transferability_results_2021-12-12_17-38-38_roberta-large--2021-12-12_19-08-47_roberta-base-records.pkl',
    'trans-v_rb-b:v_rb-lg': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/transferability_results_2021-12-12_19-08-47_roberta-base--2021-12-12_17-38-38_roberta-large-records.pkl',
    # Adv retraining
    'retrain-v_rb-lg:adv_before': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
    'retrain-v_rb-lg:adv_after': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain_v1/test-adv-records_epoch100.pkl',
    'retrain-v_rb-lg:orig_before': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/test_results.json',
    'retrain-v_rb-lg:orig_after': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain_v1/test-orig-records_epoch100.pkl',     # TODO update
    'retrain-v_rb-b:adv_before': 'bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/test_results-records.pkl',
    'retrain-v_rb-b:adv_after': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base_retrain_v1/test-adv-records_epoch100.pkl',
    'retrain-v_rb-b:orig_before': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/test_results.json',
    'retrain-v_rb-b:orig_after': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base_retrain_v1/test-orig-records_epoch100.pkl',       # TODO update
}

ref_paths = {
    'v_rb-lg': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-b': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    # 'v_rb-lg:a_hotflip': 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05.pkl',
    # 'v_rb-lg:a_textfooler': 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05_reevaled.pkl',
    # 'v_rb-lg:a_rb-b': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    # 'v_rb-lg:a_rb-b': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-lg_retrain_orig': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-lg_retrain_adv': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-lg_retrain_aug': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'trans-v_rb-lg:v_rb-b': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
    'trans-v_rb-b:v_rb-lg': 'bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/test_results-records.pkl',
}


def display_victim_results():
    dfs = []
    for name in ['v_rb-lg', 'v_rb-b']:
        path = paths[name]
        results = json.load(open(path, 'r'))
        preds = results['predictions']

        data_path = ref_paths[name]
        data = [json.loads(jline) for jline in open(data_path, 'r').read().splitlines()]
        questions = [q for x in data for q in x['questions']]
        q2dep = {q['id']: q['meta']['QDep'] for q in questions}
        q2len = {q['id']: 1 if q['meta']['QLen'] == '' else q['meta']['QLen'] for q in questions}

        mod_preds = []
        for pred in preds:
            id = pred['id']
            mod_preds.append({**pred, **{'q_depth': q2dep[id], 'q_length': q2len[id]}})

        df = pd.DataFrame.from_records(mod_preds)
        df['q_depth'] = df['q_depth'].astype(int).apply(lambda x: min(x, 5))
        df['q_depth'] = df['q_depth'].astype(str).apply(lambda x: x.replace('5', '$\geq$5'))
        df_dep = df.groupby('q_depth').agg({'is_correct': 'mean', 'q_depth': 'count'}).rename(columns={'q_depth':'N'})
        df_dep.loc['Total'] = results['EM']
        print(df_dep)

        df['q_length'] = df['q_length'].astype(int).apply(lambda x: min(x, 10))
        df['q_length'] = df['q_length'].astype(str).apply(lambda x: x.replace('10', '$\geq$10'))
        df_len = df.groupby('q_length').agg({'is_correct': 'mean', 'q_depth': 'count'}).rename(columns={'q_depth':'N'})
        df_len.loc['Total'] = results['EM']
        print(df_len)

        dfs.append(df_dep)
    
    output_df = pd.concat(dfs, axis=1)
    output_df = output_df.iloc[:,:3]
    output_df.columns = ['roberta-lg', 'N', 'roberta-base']
    output_df = output_df[['roberta-lg', 'roberta-base', 'N']]
    output_df.loc['Total', 'N'] = output_df.loc[output_df.index!='Total', 'N'].sum()
    output_df['Rb-Lg'] = output_df['roberta-lg'].apply(lambda x: f'{100*x:.1f}')
    output_df['Rb-Base'] = output_df['roberta-base'].apply(lambda x: f'{100*x:.1f}')
    output_df['N'] = output_df['N'].apply(lambda x: f'{x:.0f}')

    print(output_df.to_latex())


def display_attacker_results():
    summaries = []
    cols = ['success_rate', 'f1', 'name']

    # Attacker model performance (incl. random baselines)
    for name in ['v_rb-lg:a_rb-b', 'v_rb-b:a_rb-b', 'v_rb-lg:a_random', 'v_rb-lg:a_wordscore']:
        path = paths[name]
        df = load_as_df(path)
        df = compute_features(df, 1, 1)
        df['orig_proof_depth'] = df['orig_proof_depth'].astype(int).apply(lambda x: min(x, 5))
        df['orig_proof_depth'] = df['orig_proof_depth'].astype(str).apply(lambda x: x.replace('5', '$\geq$5'))
        summary = df[['qa_fooled', 'sampled_f1']].mean()
        summary.loc['name'] = name
        tmp_df = pd.DataFrame(summary).transpose()
        tmp_df.columns = cols
        summaries.append(tmp_df)
        continue

    # Baseline performance
    for name in ['v_rb-lg:a_textfooler']: #['v_rb-lg:a_hotflip']: #, 'v_rb-lg:a_textfooler']:
        path = paths[name]
        df = load_as_df(path)
        df['orig_sentences'] = df['context'].apply(lambda x: len(x.split('.'))-1)
        df['f1'] = 1 - df['subs'] / df['orig_sentences']
        summary = df[['adv_result', 'f1']].mean()
        summary.loc['name'] = name + '-unadj'
        tmp_df = pd.DataFrame(summary).transpose()
        tmp_df.columns = cols
        summaries.append(tmp_df)

        # filter out samples on which the Problog solver failed
        df = df[df.mod_label.isin([True, False])]
        df['mod_correct'] = df['adv_result'] & df['label']==df['mod_label']
        summary = df[['mod_correct', 'f1']].mean()
        summary.loc['name'] = name + '-adj'
        tmp_df = pd.DataFrame(summary).transpose()
        tmp_df.columns = cols
        summaries.append(tmp_df)
        continue

    output_df = pd.concat(summaries, axis=0)
    output_df.set_index('name', inplace=True)
    output_df['success_rate'] = output_df['success_rate'].apply(lambda x: f'{100*x:.1f}')
    output_df['f1'] = output_df['f1'].apply(lambda x: f'{x:.3f}')
    print(output_df.to_latex())


def display_adversarial_retraining_results():
    summaries = []
    for name in ['v_rb-lg_retrain_orig', 'v_rb-lg_retrain_adv']:
        path = paths[name]
        preds = load_as_df(path)

        data_path = ref_paths[name]
        data = [json.loads(jline) for jline in open(data_path, 'r').read().splitlines()]
        questions = [q for x in data for q in x['questions']]
        q2dep = {q['id']: q['meta']['QDep'] for q in questions}
        q2len = {q['id']: 1 if q['meta']['QLen'] == '' else q['meta']['QLen'] for q in questions}

        mod_preds = []
        for pred in preds.to_dict('records'):
            id = pred['id']
            if id.startswith('Adv-'):
                id = '-'.join(id.split('-')[1:-1])
            mod_preds.append({**pred, **{'q_depth': q2dep[id], 'q_length': q2len[id]}})

        df = pd.DataFrame.from_records(mod_preds)
        df['q_depth'] = df['q_depth'].astype(int).apply(lambda x: min(x, 5))
        df['q_depth'] = df['q_depth'].astype(str).apply(lambda x: x.replace('5', '$\geq$5'))

        summary = df[['is_correct']].mean()
        summary.loc['name'] = name
        tmp_df = pd.DataFrame(summary).transpose()
        # tmp_df.columns = cols
        summaries.append(tmp_df)

    output_df = pd.concat(summaries, axis=0)
    output_df.set_index('name', inplace=True)
    output_df['is_correct'] = output_df['is_correct'].apply(lambda x: f'{100*x:.1f}')
    print(output_df.to_latex())


def display_transferability_results():
    summaries = []
    cols = ['name', 'cnt_qafooled', 'ASR', 'acc', 'cnt']
    for name in ['trans-v_rb-lg:v_rb-b', 'trans-v_rb-b:v_rb-lg']:
        path = paths[name]
        preds = load_as_df(path)

        data_path = ref_paths[name]
        ref = load_as_df(data_path)

        preds['id'] = preds['id'].apply(lambda x: '-'.join(x.split('-')[1:-1]))
        df = pd.merge(preds, ref[['orig_proof_depth','qa_fooled','id']], on='id')
        tn, fp, fn, tp = confusion_matrix(df['is_correct'], df['qa_fooled']).ravel().tolist()
        precision = tp / (tp + fp)
        n_qafooled = df.qa_fooled.tolist().count(True)
        row = [n_qafooled, 100*(1-precision), 100*df.is_correct.mean(), len(df)]
        tmp_df = pd.DataFrame([[name, *row]], columns=cols)
        summaries.append(tmp_df)
        df_ = df[df.qa_fooled]
        df_.is_correct.mean()


    output_df = pd.concat(summaries, axis=0)
    output_df.set_index('name', inplace=True)
    print(output_df.to_latex(float_format="%.1f"))


def display_retraining_results():
    summaries = []
    cols = ['name', 'acc']
    for name in ['retrain-v_rb-lg:adv_before', 'retrain-v_rb-b:adv_before']:
        path = paths[name]
        preds = load_as_df(path)
        acc = 100*(1 - preds.qa_fooled.mean())
        summaries.append([name, acc])
        continue

    for name in ['retrain-v_rb-lg:adv_after', 'retrain-v_rb-b:adv_after', 'retrain-v_rb-lg:orig_after', 'retrain-v_rb-b:orig_after']:
        path = paths[name]
        preds = load_as_df(path)
        acc = 100*preds.is_correct.mean()
        summaries.append([name, acc])
        continue

    for name in ['retrain-v_rb-lg:orig_before', 'retrain-v_rb-b:orig_before']:
        path = paths[name]
        results = json.load(open(path, 'r'))
        preds = pd.DataFrame(results['predictions'])

        acc = 100*preds.is_correct.mean()
        summaries.append([name, acc])
        continue

    output_df = pd.DataFrame(summaries, columns=cols)
    output_df.sort_values('name', inplace=True)
    output_df.set_index('name', inplace=True)
    print(output_df.to_latex(float_format="%.1f"))


if __name__ == '__main__':
    # display_victim_results()
    # display_attacker_results()
    # display_adversarial_retraining_results()
    # display_transferability_results()
    display_retraining_results()
