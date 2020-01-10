import glob
import os
from PIL import Image
from .dataset_new import Dataset

class FolderImageDataset(Dataset):
    def __init__(self, folder_path, transforms=None):
        folder_path = os.path.expanduser(folder_path)
        self._root = folder_path
        self.images = sorted(glob.glob("{}/*.*".format(folder_path)))
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = self.images[index]
        image = Image.open(im_path)

        if self.transforms is not None:
            image = self.transforms(image)

        inputs = {}
        inputs['data'] = image

        targets = None

        return inputs, targets

    def __len__(self):
        return len(self.images)

