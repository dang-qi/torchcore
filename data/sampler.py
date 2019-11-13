import random
from math import ceil
class BaseSampler(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(BaseSampler):
    def __init__(self, dataset):
        super(SequentialSampler, self).__init__(dataset)

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

class RandomSampler(BaseSampler):
    def __init__(self, dataset, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(dataset)
        if replacement and num_samples is None:
            raise ValueError('num_sampler should be specified when replacement is enabled')

        if not replacement and num_samples is not None:
            raise ValueError('num_sampler should not be specified when replacement is not enabled')
        self._replacement = replacement
        self._num_sample = num_samples

    def __iter__(self):
        n = len(self.dataset)
        if self._replacement:
            return iter(random.choice(range(n)) for _ in range(self._num_sample))
        else:
            inds = list(range(n))
            random.shuffle(inds)
            return iter(inds)

class BatchSampler(BaseSampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch=[]
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return ceil(len(self.sampler) / self.batch_size)
        
