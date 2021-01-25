import numpy as np
from torch.utils.data.sampler import Sampler, SequentialSampler

class MixBatchSampler(Sampler):
    def __init__(self, dataset_len, mode, batch_size, shuffle=False, drop_last=True):
        self.dataset_len = dataset_len
        self.group_num = len(dataset_len)
        self.mode = mode
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        index_group = self.gen_index_group()
        self.sampler = self.get_sampler(index_group)


    def __iter__(self):
        if self.shuffle:
            self.shuffle_once()
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
    def shuffle_once(self):
        index_group = self.gen_index_group()
        self.sampler = self.get_sampler(index_group)

    def gen_index_group(self):
        if self.mode=='balance':
            if self.batch_size % self.group_num != 0:
                raise ValueError('The number of dataset should be divide by batch size in balance mode')
            max_len = max(self.dataset_len)
            index_group = []
            totol_data = 0
            for data_len in self.dataset_len:
                inds = np.arange(data_len)
                inds = inds + totol_data
                if self.shuffle:
                    np.random.shuffle(inds)
                n_times = max_len // data_len
                rest_num = max_len % data_len
                inds_all = []
                for _ in range(n_times):
                    if self.shuffle:
                        np.random.shuffle(inds)
                    inds_all.append(inds.copy())
                if rest_num > 0:
                    if self.shuffle:
                        np.random.shuffle(inds)
                    inds_all.append(inds[:rest_num])
                inds = np.concatenate(inds_all)

                #rest = inds[:rest_num]
                #inds = np.tile(inds, n_times)
                #inds = np.concatenate((inds, rest))
                index_group.append(inds)
                totol_data += data_len
            return index_group
        elif self.mode == 'single':
            data_len = self.dataset_len[0]
            index_group = []
            inds = np.arange(data_len)
            if self.shuffle:
                np.random.shuffle(inds)
            index_group.append(inds)
            return index_group
        else:
            raise ValueError('Unknow mode: {}'.format(self.mode))

    def get_sampler(self, index_group):
        sampler = []
        ind_lens = [len(index) for index in index_group]
        max_len = max(ind_lens)
        i = 0
        batch = []
        while True:
            for j, index in enumerate(index_group):
                if i >= ind_lens[j]:
                    continue
                batch.append(index[i])
                if len(batch) == self.batch_size:
                    #print(batch)
                    batch.sort()
                    sampler.extend(batch)
                    batch = []
            i += 1
            if i >= max_len:
                break
        return sampler


