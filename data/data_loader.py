from .data_fecher import _MapDatasetFetcher
from .sampler import RandomSampler, SequentialSampler, BatchSampler
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, collate_fn=None, drop_last=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        # The Iterable dataset is not considered for now

        if sampler is not None and shuffle:
            raise ValueError('sampler and shuffle are mutual exclusive!')

        if batch_sampler is not None:
            if sampler or batch_size != 1 or shuffle or drop_last:
                raise ValueError('batch_sampler is mutual exclusive with '
                                 'batch size, shuffle, sampler and drop last')
            batch_size = None
            drop_last=False

        if sampler is None: # get the default sampler
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        #TODO:set default collate_fn
        if collate_fn is None:
            pass
            #collate_fn = defalt_collate_fn
        
        self.collate_fn = collate_fn

    @property
    def _index_sampler(self):
        '''Return the index sampler'''
        if self.batch_sampler is not None:
            return self.batch_sampler
        else:
            return self.sampler

    def __iter__(self):
        return _SingleProcessDataLoaderIter(self)

    def __len__(self):
        return len(self._index_sampler)

class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter) # May raise StopIteration

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._index_sampler)

class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)

        # TODO: find out why is there a data fecher
        self._datafecher = _MapDatasetFetcher(self._dataset, self._collate_fn, self._drop_last)

    def __next__(self):
        index = self._next_index() # May raise StopIteration
        data = self._datafecher.fetch(index) # May raise StopIteration

        return data


    