from random import shuffle
from torch.utils import data

from allennlp.data.samplers import SequentialSampler, Sampler, BasicBatchSampler, BatchSampler
from .utils import flatten_list, lfilter

@Sampler.register("custom_sequential")
class CustomSequentialSampler(SequentialSampler):
    ''' Allows sampling based on the length of the proof
        (QLen).
    '''
    def __init__(self, data_source: data.Dataset):
        super().__init__(data_source)
        self._partition(data_source)
        self.samples = []       # Placeholder which gets filled at a later stage

    def _partition(self, data_source):
        ''' Obtain lists of indices for samples by 
            QLen
        '''
        self.QLens = {}
        for n,d in enumerate(data_source):
            qlen = d.fields['metadata'].metadata['QLen']
            if qlen in self.QLens:
                self.QLens[qlen].append(n)
            else:
                self.QLens[qlen] = [n]


@BatchSampler.register("custom")
class CustomBasicBatchSampler(BasicBatchSampler):
    ''' Wraps the CustomSequentialSampler so it 
        samples batches.
    '''
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
        self.req_QLens = []
        self.batch = []
        self.batches = []
        self._mode = 'retrieval'

    def __iter__(self):
        batch = []
        for idx in self.get_samples():
            batch.append(idx)
            if len(batch) == self.batch_size:
                self._save_batch(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            self._save_batch(batch)
            yield batch
            batch = []

    def _save_batch(self, batch):
        if self._mode == 'retrieval':
            self.batch.extend(batch)
            self.batches.extend(batch)

    def get_samples(self):
        if self._mode == 'retrieval':
            # return self.sampler.QLens[self.QLen]
            ids = flatten_list([self.sampler.QLens[k] for k in self.req_QLens if k in self.sampler.QLens])
            shuffle(ids)
            return ids
        elif self._mode == 'binary_classification':
            return self.sampler.samples
        else:
            raise NotImplementedError

    def set_mode(self, mode: str):
        assert mode in ['binary_classification', 'retrieval']
        self._mode = mode
