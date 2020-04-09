import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset
from ..transforms import ToTensor, Normalize, Compose

class DemoDataset(Dataset):
    def __init__( self, path_list, transforms=None):
        self.path_list = path_list
        self._transforms = transforms

    def __getitem__(self, idx):
        # Load image
        img_path = self.path_list[idx]
        img = Image.open(img_path).convert('RGB')
        #ori_image = img.copy()

        inputs = {}
        inputs['data'] = img


        # for debug
        inputs['ori_image'] = np.asarray(inputs['data'].copy())

        transforms_post = Compose([ToTensor(), Normalize()])
        inputs, _ = transforms_post(inputs )
        return inputs

    def __len__(self):
        return len(self.path_list)
