import numpy as np

from .dataset_new import Dataset

class MixDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, shuffle=False, sample=False, transforms_a=None, transforms_b=None):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.transforms_a = transforms_a
        self.transforms_b = transforms_b
        len_a = len(dataset_a)
        len_b = len(dataset_b)
        if sample is None:
            self.dev_a = len_a
            self.dev_all = len_a + len_b
            self.inds = np.arange(self.dev_all)
        elif sample == 'subsample':
            self.dev_a = min(len_a, len_b) # devide for dataset a
            self.dev_all = self.dev_a * 2
            self.inds = np.arange(self.dev_all)
        elif sample == 'duplicate':
            self.dev_a = len_a
            if len_a < len_b:
                times = len_b // len_a
                extra = len_b % len_a
                ind_a = np.tile(np.arange(len_a), times)
                ind_a_extra = np.random.choice(np.arange(len_a), extra)
                ind_a = np.concatenate((ind_a, ind_a_extra))
                ind_b = np.arange(len_a, len_a+len_b)
            else:
                times = len_a // len_b
                extra = len_a % len_b
                ind_a = np.arange(len_a)
                ind_b = np.tile(np.arange(len_a, len_a+len_b), times)
                ind_b_extra = np.random.choice(np.arange(len_a, len_a+len_b), extra)
                ind_b = np.concatenate((ind_b, ind_b_extra))
            self.inds = np.concatenate((ind_a, ind_b))
        else:
            raise ValueError('Wrong sample method, it can be None, subsample or duplicate')

        if shuffle:
            np.random.shuffle(self.inds)


    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        ind = self.inds[idx]
        if ind <self.dev_a:
            inputs, targets = self.dataset_a[ind]
            if self.transforms_a is not None:
                inputs, targets = self.transforms_a(inputs, targets)
        else:
            inputs, targets = self.dataset_b[ind-self.dev_a]
            if self.transforms_b is not None:
                inputs, targets = self.transforms_b(inputs, targets)
        return inputs, targets
