import numpy as np

from .dataset_new import Dataset

class MultiMixDataset(Dataset):
    def __init__(self, datasets, shuffle=False, sample=None, add_dataset_label=False):
        '''
            datasets: list[dataset1, dataset2 ...]
            If transforms needed, please do it on seperate datasets
        '''
        self.datasets = datasets
        #self.transforms = transforms
        self.add_dataset_label = add_dataset_label
        if sample is None:
            self.dataset_bins=[]
            len_all = 0
            for dataset in datasets:
                len_all += len(dataset)
                self.dataset_bins.append(len_all)
            self.dataset_bins = np.array(self.dataset_bins)
            self.inds = np.arange(len_all)
        else:
            raise ValueError('Wrong sample method, it can be None, subsample or duplicate')

        # DO NOT SHUFFLE it if you need to use sampler
        if shuffle:
            np.random.shuffle(self.inds)


    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        idx = self.inds[idx]
        ind = np.digitize(idx, self.dataset_bins)
        dataset = self.datasets[ind]
        dataset_ind = idx - self.dataset_bins[ind]
        inputs, targets = dataset[dataset_ind]
        if self.add_dataset_label:
            inputs['dataset_label'] = ind
        return inputs, targets
