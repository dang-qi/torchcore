
class _BaseDatasetFetcher(object):
    def __init__(self, dataset, collate_fn, drop_last):
        self.dataset = dataset
        #self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, index):
        raise NotImplementedError()

class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset,  collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        data = [self.dataset[idx] for idx in possibly_batched_index]
        #if self.auto_collation:
        #    data = [self.dataset[idx] for idx in possibly_batched_index]
        #else:
        #    data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)