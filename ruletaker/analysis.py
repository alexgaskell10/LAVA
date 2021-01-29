import json
import os
import sys
import glob

class ResultsProcessor():
    def __init__(self, dir):
        self.dir = dir
        self.config_path = os.path.join(dir, 'config.json')

        self.load_best_metrics()
        self.add_proof_depth()

    def load_best_metrics(self, metric='validation_EM'):
        ''' Add proof depth for each prediction
        '''
        metrics_paths = glob.glob(os.path.join(self.dir, 'metrics_epoch_*.json'))
        # metrics_paths = glob.glob(os.path.join(self.dir, 'last_epoch_validation*.json'))

        self.scores = {}
        best_score, best_idx = 0, None
        for path in metrics_paths:
            with open(path, 'r') as f:
                self.scores[path] = json.load(f)
            
            if self.scores[path][metric] > best_score:
                self.best_score = self.scores[path][metric]
                self.best_idx = path

    def add_proof_depth(self):
        ''' Add proof depth for each prediction
        '''
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        data_path = config['validation_data_path']
        
        # Extract {id: {qid: depth, ...}} pairs from original dev data
        id2depth = {}
        for line in open(data_path).readlines():
            row = json.loads(line)
            id = row.pop('id')
            id2depth[id] = {}
            for q in row['questions']:
                id2depth[id][q.pop('id')] = q.pop('meta').pop('QDep')

        # Add proof depth for each prediction
        # There are multiple predictions per question (one per epoch)
        preds = self.scores[self.best_idx]['validation_predictions']
        self.preds = {}
        for pred in preds:
            qid = pred['id']
            id = '-'.join(qid.split('-')[:-1])
            if id in id2depth:
                pred['QDep'] = id2depth[id][qid]
                self.preds[qid] = pred


class ResultsAnalyzer(ResultsProcessor):
    def __init__(self, *args):
        super().__init__(*args)
        self.by_depth()

    def by_depth(self):
        ''' Analyze performance by question depth.
        '''
        # {depth: [#correct, #samples]}
        scores = {i: [0, 0] for i in range(10)}
        for id, pred in self.preds.items():
            depth = pred['QDep']
            scores[depth][0] += pred['is_correct']
            scores[depth][1] += 1

        [print(k,v[1],v[0]/v[1]) for k,v in scores.items() if v[1] > 0]
        print(sum([v[0] for k,v in scores.items() if k <= 5]) / sum([v[1] for k,v in scores.items() if k <= 5]))


if __name__ == '__main__':
    os.chdir('ruletaker')
    ResultsAnalyzer('runs/depth-3ext')
