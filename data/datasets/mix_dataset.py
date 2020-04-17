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
        if sample:
            self.dev_a = min(len_a, len_b) # devide for dataset a
            self.dev_all = self.dev_a * 2
        else:
            self.dev_ind = len_a
            self.dev_all = len_a + len_b

        self.inds = np.arange(self.dev_all)
        if shuffle:
            np.random.shuffle(self.inds)


    def __len__(self):
        return self.dev_all

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
