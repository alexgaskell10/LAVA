import json, os, sys
import pandas as pd

paths = {
    'ruletaker-lg': 'bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/test_results.json',
    'ruletaker-base': 'bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/test_results.json',
}

reference_paths = {
    'ruletaker-lg': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
    'ruletaker-base': 'data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl',
}


def display_ruletaker_results():
    dfs = []
    for name in ['ruletaker-lg', 'ruletaker-base']:
        path = paths[name]
        results = json.load(open(path, 'r'))
        preds = results['predictions']

        ref_path = reference_paths[name]
        data = [json.loads(jline) for jline in open(ref_path, 'r').read().splitlines()]
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


if __name__ == '__main__':
    display_ruletaker_results()
