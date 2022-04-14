import os
import pandas as pd
from .dataset_new import Dataset
from ..transforms import Compose
from ..transforms.build import build_transform
class HMRecommendationDataset(Dataset):
    def __init__(self, root_path, transforms=None):
        self.root_path = root_path
        self.article_path = os.path.join(root_path, 'articles.csv')
        self.customer_path = os.path.join(root_path, 'customers.csv')
        self.article_path = os.path.join(root_path, 'transactions_train.csv')
        if transforms is None:
            transforms = []
        self._transforms=Compose(transforms)

        self.articles = pd.read_csv(self.article_path)
        self.add_image_path()
        self.remove_empty_path()
        
    def remove_empty_path(self):
        self.articles = self.articles[self.articles['path'].apply(os.path.exists)]

    def add_image_path(self):
        im_path = []
        base_path = 'images'
        for _,a in self.articles.iterrows():
            folder = a['article_id'][:3]
            file_name = a['article_id'] + '.jpg'
            path = os.path.join(self.root_path,base_path, folder, file_name)
            im_path.append(path)
        self.articles['path'] = im_path

    def __getitem__(self, idx):
        path = self.articles['path'][idx]
        article_id = self.articles['article_id']
        img = self._load_image(path)
        inputs = {}
        inputs['data'] = img
        inputs['article_id'] = article_id
        targets = None
        if self._transforms is not None:
            for t in self._transforms:
                inputs, targets = t(inputs, targets)
        return inputs, targets