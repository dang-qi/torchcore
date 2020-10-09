from PIL import Image
import cv2
import torch
from .dataset_new import Dataset

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video_path, transforms=None ):
        self.video_path = video_path
        self.vidcap = cv2.VideoCapture(video_path)
        self.transforms = transforms

    def __iter__(self):
        return VideoDatasetIterator(self)


    class VideoDatasetIterator():
        def __init__(self, video_dataset):
            self.cap = cv2.VideoCapture(video_dataset.video_path)
            self.transforms = video_dataset.transforms

        def __next__(self):
            ...