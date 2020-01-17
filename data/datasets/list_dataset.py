from PIL import Image
from .dataset_new import Dataset

class ListDataset(Dataset):
    def __init__(self, list_path, transforms=None ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')
        inputs = {}
        inputs['path'] = img_path
        inputs['ori_image'] = img
        targets = None

        if self.transforms is not None:
            inputs['data'],  targets = self.transforms(img, targets)
        else:
            inputs['data'] = img

        return inputs, targets

    def __len__(self):
        return len(self.img_files)