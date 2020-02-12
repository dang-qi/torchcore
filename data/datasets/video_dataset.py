from PIL import Image
import cv2
import torch
from .dataset_new import Dataset

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video_path, transforms=None ):
        self.vidcap = cv2.VideoCapture(video_path)
        self.transforms = transforms

    def __iter__(self)
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