from torch.utils.data import Dataset
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, start=0, end=100) -> None:
        self.data = np.array(np.arange(start, end))

    def __getitem__(self, index):
        inputs = {}
        targets = {}
        inputs['index'] = index
        targets['data'] = self.data[index]
        return inputs, targets

    def __len__(self) -> int:
        return len(self.data)

    
