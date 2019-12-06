from .data_fecher import _MapDatasetFetcher
from .sampler import RandomSampler, SequentialSampler, BatchSampler
class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    no iterable-style datasets with single- or no multi-process loading, customizing
    loading order and optional automatic batching (collation) and no memory pinning.
    See :py:mod:`torch.utils.data` documentation page for more details.
    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, :attr:`shuffle` must be ``False``. The sampler
            should have __iter__ method and the iterator should yield an index number 
            each time.
        batch_sampler (Sampler, optional): like :attr:`sampler`, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
            The batch sampler should have __iter__ method and return a list of index 
            each time.
        no num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        no pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        no worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.
    .. note:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
              When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
              an infinite sampler is used, whose :meth:`__len__` is not
              implemented, because the actual length depends on both the
              iterable as well as multi-process loading configurations. So one
              should not query this method unless they work with a map-style
              dataset. See `Dataset Types`_ for more details on these two types
              of datasets.
    """
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


    