import json, os, sys
import pickle as pkl
import pandas as pd
from glob import glob
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from analysis import load_as_df, compute_features

paths = {
    'v_rb-lg': 'bin/runs/ruletaker/2021-12-20_10-48-09_roberta-large/test_results.json',
    'v_rb-b': 'bin/runs/ruletaker/2021-12-21_08-55-58_roberta-base/test_results.json',
    'v_rb-lg:a_rb-b': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
    'v_rb-b:a_rb-b': 'bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/test_results-records.pkl',
    # Baselines
    'v_rb-lg:a_hotflip': 'bin/runs/baselines/hotflip/2021-12-16_11-00-14_reevaled.pkl',     # TODO
    'v_rb-lg:a_textfooler': 'bin/runs/baselines/textfooler/2021-12-16_11-00-14_reevaled.pkl',       # TODO
    # Random benchmarks
    'v_rb-lg:a_random': 'bin/runs/baselines/random_adversarial/random_2021-12-22_14-02-58_results-records.pkl',
    'v_rb-lg:a_wordscore': 'bin/runs/baselines/random_adversarial/word_score_2021-12-22_14-02-58_results-records.pkl',
    # Adversarial retraining
    'v_rb-lg_retrain_orig': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain/test-orig-records_epoch100.pkl',
    'v_rb-lg_retrain_adv': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large_retrain/test-adv-records_epoch100.pkl',
    # Transferability
    'trans-v_rb-lg:v_rb-b': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/transferability_results_2021-12-12_17-38-38_roberta-large--2021-12-12_19-08-47_roberta-base-2-records.pkl',
    'trans-v_rb-b:v_rb-lg': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/transferability_results_2021-12-12_19-08-47_roberta-base--2021-12-12_17-38-38_roberta-large-2-records.pkl',
    'trans-hotflip-v_rb-lg:v_rb-b': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/transferability_results_rb_lg--rb_base_hotflip-records.pkl',
    'trans-textfooler-v_rb-lg:v_rb-b': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/transferability_results_rb_lg--rb_base_hotflip-records.pkl',
    # Adv retraining
    'retrain-v_rb-lg:adv_before': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
    'retrain-v_rb-lg:adv_after': 'bin/runs/ruletaker/2021-12-20_10-48-09_roberta-large_retrain_v2/test-adv-records_epoch100.pkl',
    'retrain-v_rb-lg:orig_before': 'bin/runs/ruletaker/2021-12-20_10-48-09_roberta-large/test_results.json',
    'retrain-v_rb-lg:orig_after': 'bin/runs/ruletaker/2021-12-20_10-48-09_roberta-large_retrain_v2/test-orig-records_epoch100.pkl',     # TODO maybe update
    'retrain-v_rb-b:adv_before': 'bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/test_results-records.pkl',
    'retrain-v_rb-b:adv_after': 'bin/runs/ruletaker/2021-12-21_08-55-58_roberta-base_retrain_v1/test-adv-records_epoch100.pkl',
    'retrain-v_rb-b:orig_before': 'bin/runs/ruletaker/2021-12-21_08-55-58_roberta-base/test_results.json',
    'retrain-v_rb-b:orig_after': 'bin/runs/ruletaker/2021-12-21_08-55-58_roberta-base_retrain_v1/test-orig-records_epoch100.pkl',
    # Analysis
    'v_rb-lg:a_rb-b:ana': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
}

dirs = {
    'num_perturbs': [
        'bin/runs/num_perturbs/2021-12-16_10-41-53',
        'bin/runs/num_perturbs/2021-12-16_10-42-44',
        'bin/runs/num_perturbs/2021-12-16_10-43-16',
        'bin/runs/num_perturbs/2021-12-16_10-43-27',
        'bin/runs/num_perturbs/2021-12-16_10-43-41',
    ],
    'val_mc': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large'
}

ref_paths = {
    'v_rb-lg': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-b': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    # 'v_rb-lg:a_hotflip': 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05.pkl',
    # 'v_rb-lg:a_textfooler': 'bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05_reevaled.pkl',
    'v_rb-lg:a_rb-b': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-lg_retrain_orig': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-lg_retrain_adv': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'v_rb-lg_retrain_aug': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'trans-v_rb-lg:v_rb-b': 'bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/test_results-records.pkl',
    'trans-v_rb-b:v_rb-lg': 'bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/test_results-records.pkl',
    'trans-hotflip-v_rb-lg:v_rb-b': 'bin/runs/baselines/hotflip/2021-12-16_11-00-14_reevaled_bu.pkl',
    'trans-textfooler-v_rb-lg:v_rb-b': 'bin/runs/baselines/textfooler/2021-12-16_11-00-14_reevaled_bu.pkl',
    'v_rb-lg:a_rb-b:ana': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/test_results.json',
}

def compute_f1(row):
    result, context = row
    if result is not None:
        unchanged = [r.replace(' ','')==c.replace(' ','') for r,c in zip(result.split('.')[:-1], context.split('.')[:-1])]
        return sum(unchanged) / len(unchanged)
    return None


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
        # df['q_depth'] = df['q_depth'].astype(str).apply(lambda x: x.replace('5', '$\geq$5'))
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
    for name in ['v_rb-lg:a_hotflip', 'v_rb-lg:a_textfooler']:
        path = paths[name]
        df = load_as_df(path)
        df['orig_sentences'] = df['context'].apply(lambda x: len(x.split('.'))-1)
        
        # df['f1'] = 1 - df['subs'] / df['orig_sentences']
        df['f1_'] = df[['result','context']].apply(compute_f1, axis=1)
        df['f1'] = df[~df['f1_'].isnull()]['f1_'].mean()

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
        precision = tp / (tp + fp)      #3989 2112 10499 3592
        n_qafooled = df.qa_fooled.tolist().count(True)
        row = [n_qafooled, 100*(1-precision), 100*df.is_correct.mean(), len(df)]
        tmp_df = pd.DataFrame([[name, *row]], columns=cols)
        summaries.append(tmp_df)
        # df_ = df[df.qa_fooled]
        # df_.is_correct.mean()

    # for name in ['trans-hotflip-v_rb-lg:v_rb-b', 'trans-textfooler-v_rb-lg:v_rb-b']:
    #     path = paths[name]
    #     preds = load_as_df(path)

    #     data_path = ref_paths[name]
    #     ref = load_as_df(data_path)
    #     ref['qa_fooled'] = ref['adv_result'] & ref['label']==ref['mod_label']

    #     preds['id'] = preds['id'].apply(lambda x: '-'.join(x.split('-')[1:-1]))
    #     df = pd.merge(preds, ref[['QDep','qa_fooled','id']], on='id')
    #     tn, fp, fn, tp = confusion_matrix(df['is_correct'], df['qa_fooled']).ravel().tolist()
    #     precision = tp / (tp + fp)
    #     n_qafooled = df.qa_fooled.tolist().count(True)
    #     row = [n_qafooled, 100*(1-precision), 100*df.is_correct.mean(), len(df)]
    #     tmp_df = pd.DataFrame([[name, *row]], columns=cols)
    #     summaries.append(tmp_df)

    output_df = pd.concat(summaries, axis=0)
    output_df.set_index('name', inplace=True)
    print(output_df.to_latex(float_format="%.1f"))


def display_retraining_results():
    summaries, scores = [], []
    cols = ['name', 'acc']
    for name in ['retrain-v_rb-lg:adv_before', 'retrain-v_rb-b:adv_before']:
        path = paths[name]
        preds = load_as_df(path)
        acc = 100*(1 - preds.qa_fooled.mean())
        summaries.append([name, acc])
        scores.append(1 - preds.qa_fooled)
        continue

    for name in ['retrain-v_rb-lg:adv_after', 'retrain-v_rb-b:adv_after', 'retrain-v_rb-lg:orig_after', 'retrain-v_rb-b:orig_after']:
        path = paths[name]
        preds = load_as_df(path)
        acc = 100*preds.is_correct.mean()
        summaries.append([name, acc])
        scores.append(preds.is_correct)
        continue

    for name in ['retrain-v_rb-lg:orig_before', 'retrain-v_rb-b:orig_before']:
        path = paths[name]
        results = json.load(open(path, 'r'))
        preds = pd.DataFrame(results['predictions'])
        acc = 100*preds.is_correct.mean()
        summaries.append([name, acc])
        scores.append(preds.is_correct)
        continue

    names = {
        'retrain-v_rb-b': 'rb-base', 'retrain-v_rb-lg': 'rb-lg',
        'adv': 'Adv.', 'orig': 'Orig.', 
        'after': 'After', 'before': 'Before'
    }
    output_df = pd.DataFrame(summaries, columns=cols)
    output_df.sort_values('name', inplace=True)
    output_df[['name','tmp']] = output_df['name'].str.split(':', expand=True)
    output_df[['data','tmp']] = output_df['tmp'].str.split('_', expand=True)
    # output_df['name'] = output_df['name'].apply(lambda x: names[x])
    output_df[['Victim', '', 'DSet']] = output_df[['name', 'tmp', 'data']].applymap(lambda x: names[x])
    output_df = output_df.pivot(index=['Victim', 'DSet'], columns=[''])
    output_df = output_df[[('acc', 'Before'), ('acc',  'After')]]
    print(output_df.to_latex(float_format="%.2f"))

    # Significance test
    for i,j in [[0,2], [1,3], [4,6], [5,7]]:
        x1 = scores[i]
        x2 = scores[j]
        print(stats.ttest_ind(x1, x2, equal_var=False))



def display_num_perturbs():
    summaries = []
    cols = ['success_rate', 'f1', 'name']

    files_dct = {}
    # Obtain files
    for dir in dirs['num_perturbs']:
        files = glob(dir+'/**/test_results-records.pkl', recursive=True)
        for file in files:
            key = file.strip(dir).split('/')[0]
            files_dct[key] = file

    for name, path in files_dct.items():
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

    output_df = pd.concat(summaries, axis=0)
    output_df['Max. SentElim'] = output_df['name'].apply(lambda x: int(x.replace('SE-','')[0]))
    output_df['Max. EquivSub'] = output_df['name'].apply(lambda x: int(x[-1]))

    asr_df = output_df[['success_rate', 'Max. SentElim', 'Max. EquivSub']].pivot(index='Max. SentElim',columns='Max. EquivSub',values='success_rate')
    f1_df = output_df[['f1', 'Max. SentElim', 'Max. EquivSub']].pivot(index='Max. SentElim',columns='Max. EquivSub',values='f1')

    asr_annots = asr_df.applymap(lambda x: f'{x:.2f}'[1:])
    f1_annots = f1_df.applymap(lambda x: f'{x:.2f}'[1:])
    # asr_df.fillna(0.5, inplace=True)
    # f1_df.fillna(0.5, inplace=True)

    sns.set(font_scale=2.)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11,6))
    sns.heatmap(asr_df, cmap="YlGnBu", ax=ax0, cbar=False, annot=asr_annots, fmt='', annot_kws={"fontsize":18})
    sns.heatmap(f1_df, cmap="YlGnBu", ax=ax1, cbar=False, annot=f1_annots, fmt='', annot_kws={"fontsize":18})
    # sns.heatmap(asr_df, annot=True, cmap="YlGnBu", ax=ax0, cbar=False, fmt=".2f")
    # sns.heatmap(f1_df, annot=True, cmap="YlGnBu", ax=ax1, cbar=False, fmt=".2f")
    ax0.set_title('ASR', fontsize=30)
    ax1.set_title('F1', fontsize=30)
    ax1.set(ylabel=None, yticklabels=[])
    fig.tight_layout()
    plt.savefig('adversarial/figs/num_perturbs.png')
    plt.clf()

    # https://stackoverflow.com/questions/33158075/custom-annotation-seaborn-heatmap    < see this for custom annotations
    pass


def display_val_monte_carlo():
    # Get paths to pkl files
    dir_name = 'val_mc'
    dir_path = dirs[dir_name]
    paths = glob(dir_path + '/val_mc*.pkl')

    results = []
    for path in paths:
        val_mc = int(path.split('/')[-1].lstrip('val_mc-').rstrip('_test_results-records.pkl'))

        df = load_as_df(path)
        asr = df.qa_fooled.mean()*100
        results.append([val_mc, asr])

    sns.set_theme()
    x, y = zip(*results)
    fig, ax0 = plt.subplots(1, 1, figsize=(3.5,3))
    sns.lineplot(x, y, ax=ax0, marker="o")
    plt.xscale('log', basex=2)
    ax0.set(ylabel='ASR', xlabel='N. Samples')
    fig.tight_layout()
    plt.savefig('adversarial/figs/val_mc.png')
    plt.clf()


def analyze_attacks():
    name = 'v_rb-lg:a_rb-b:ana'
    path = paths[name]    
    df = load_as_df(path)
    df = compute_features(df, 1, 1)
    df['orig_proof_depth'] = df['orig_proof_depth'].astype(int).apply(lambda x: min(x, 5))

    ref_path = ref_paths[name]
    ref_ = json.load(open(ref_path, 'r'))['predictions']
    ref = pd.DataFrame.from_records(ref_)
    ref = pd.merge(ref, df[['id', 'orig_proof_len', 'orig_proof_depth']], on='id')

    df = pd.merge(df, ref[['id', 'is_correct']], on='id')

    ref['polarity'] = 1 - ref['phrase'].str.contains('not')
    df['orig_polarity'] = df['polarity']
    df['polarity'] = 1 - df['sampled_question'].str.contains('not')

    df['orig_proof_len'] = df['orig_proof_len'].apply(lambda x: min(x, 5))
    df.groupby(['label','orig_polarity']).aggregate({'qa_fooled':'mean', 'is_correct': 'mean', 'id':'count'})
    df.groupby('orig_proof_depth').aggregate({'qa_fooled':'mean', 'id':'count'})
    df.groupby('orig_proof_len').aggregate({'qa_fooled':'mean', 'is_correct': 'mean', 'id':'count'})
    ref.groupby('orig_proof_len').aggregate({'is_correct':'mean', 'id':'count'})
    ref.groupby(['answer','polarity']).aggregate({'is_correct':'mean', 'id':'count'})

    df_sum = df.groupby('orig_proof_len').aggregate({'qa_fooled':'mean', 'is_correct': 'mean', 'id':'count'})
    df_sum.index = [f'$\geq$5' if x == 5 else str(x) for x in df_sum.index]
    df_sum.iloc[:,:2] = df_sum.iloc[:,:2] * 100
    print(df_sum.to_latex(float_format="%.1f"))

    df_sum = df.groupby(['label','orig_polarity']).aggregate({'qa_fooled':'mean', 'is_correct': 'mean', 'id':'count'})
    df_sum.iloc[:,:2] = df_sum.iloc[:,:2] * 100
    print(df_sum.to_latex(float_format="%.1f"))

    pass


if __name__ == '__main__':
    # display_victim_results()
    # display_attacker_results()
    # display_adversarial_retraining_results()
    # display_transferability_results()
    display_retraining_results()
    # display_num_perturbs()
    # display_val_monte_carlo()
    # analyze_attacks()
