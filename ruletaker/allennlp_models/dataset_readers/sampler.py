from torch.utils import data

from allennlp.data.samplers import SequentialSampler, Sampler, BasicBatchSampler, BatchSampler


@Sampler.register("custom_sequential")
class CustomSequentialSampler(SequentialSampler):
    def __init__(self, data_source: data.Dataset):
        super().__init__(data_source)
        self._partition(data_source)

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
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
        self.QLen = None

    def __iter__(self):
        batch = []
        for idx in self.sampler.QLens[self.QLen]:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
