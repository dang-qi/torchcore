import os
import copy
import pickle
from pycocotools.coco import COCO
from ..transforms import Compose
import numpy as np
import cv2
from PIL import Image 

class Dataset:
    '''Basic Dataset'''
    _repr_indent = 4
    def __init__(self, root=None, anno=None, part=None, transforms=None ):
        if root is not None:
            root = os.path.expanduser(root)
        self._part = part
        if anno is not None:
            anno = os.path.expanduser(anno)
            self._anno = anno
            self._load_anno()
        self._root = root
        # make it compatible
        if transforms is None:
            transforms = []
        self._transforms=Compose(transforms)
        #self._set_aspect_ratio_flag()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of data: {}".format(self.__len__())]
        if self._root is not None:
            body.append("Root location: {}".format(self._root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self._transforms is not None:
            body += [repr(self._transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return ""

    def _load_anno(self):
        with open(self._anno, 'rb') as f:
            self._images = pickle.load(f)[self._part] 

    def convert_to_xyxy(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2]+=obj['bbox'][0]
                obj['bbox'][3]+=obj['bbox'][1]

    def set_first_n_subset(self, n):
        self._images = self._images[:n]
        self._set_aspect_ratio_flag()

    def _set_aspect_ratio_flag(self):
        '''We want to put the dataset that has similar aspect ratio together,
           
           So I put the flag here. The 0 means aspect ratio less than one.
           1 means aspect ratio equal or bigger than one
        '''
        self.aspect_ratio_flag = np.zeros(len(self._images),dtype=np.uint8)
        for i, im in enumerate(self._images):
            aspect_ratio = im['height']/im['width']
            if aspect_ratio >= 1:
                self.aspect_ratio_flag[i] = 1

    def set_category_subset(self, cat_id, ignore_other_category=True):
        '''
        cat_id: int or list
        make the dataset the subset with only some of the category
        '''
        if not hasattr(self, 'category_index_dict'):
            self.generate_cat_dict()
        if isinstance(cat_id, int):
            cat_id = [cat_id]
        the_sets = [set(self.category_index_dict[i]) for i in cat_id]
        im_indexs = set.union(set(), *the_sets)

        if ignore_other_category:
            cat_id = set(cat_id)
            self._images = []
            for i in im_indexs:
                im = copy.deepcopy(self._original_images[i])
                im_objs = im['objects']
                im['objects'] = []
                for obj in im_objs:
                    if obj['category_id'] in cat_id:
                        im['objects'].append(obj)
                self._images.append(im)
        else:
            self._images = [self._original_images[i] for i in im_indexs]
        self._set_aspect_ratio_flag()

    def set_subset_by_image_id(self, image_ids):
        if not hasattr(self, '_original_images'):
            self._original_images = self._images
        self._images =[ im for im in self._original_images if im['id'] in image_ids]


    def map_category_id_to_continous(self,start_id=1):
        id_set = set()
        for image in self._images:
            for obj in image['objects']:
                id_set.add(obj['category_id'])
        ids = sorted(list(id_set))
        self.id_map = {aid:i+start_id for i, aid in enumerate(ids)}
        self.cast_id_map = {i+start_id:aid for i, aid in enumerate(ids)}
        #print(id_map)
        for image in self._images:
            for obj in image['objects']:
                obj['category_id'] = self.id_map[obj['category_id']]

    def cast_result_id(self, results,):
        if not hasattr(self, 'cast_id_map'):
            return 
        assert isinstance(results, list)
        zero_start = False
        if hasattr(self, 'zero_start_label_index'):
            if self.zero_start_label_index:
                zero_start = True
        
        for result in results:
            temp_id = result['category_id']
            if zero_start:
                result['category_id'] = self.cast_id_map[temp_id]-1
            else:
                result['category_id'] = self.cast_id_map[temp_id]


    def generate_cat_dict(self):
        if hasattr(self, 'category_index_dict'):
            print('category_index_dict has been generated')
            return

        self.category_im_dict = {}
        self.category_index_dict = {}
        if not hasattr(self, '_original_images'):
            self._original_images = self._images
        
        for i,im in enumerate(self._original_images):
            im_cat_id = set()
            for obj in im['objects']:
                cat_id = obj['category_id']
                if cat_id in im_cat_id:
                    continue
                else:
                    im_cat_id.add(cat_id)
                if cat_id not in self.category_im_dict:
                    self.category_im_dict[cat_id] = []
                    self.category_index_dict[cat_id] = []
                else:
                    self.category_im_dict[cat_id].append(im)
                    self.category_index_dict[cat_id].append(i)

    def generate_id_dict(self):
        print('generating id dict...')
        self.id_dict = dict()
        for i,im in enumerate(self._images):
            im_id = im['id']
            self.id_dict[im_id] = i

    def extract_by_id(self, im_id):
        if not hasattr(self,'id_dict'):
            self.generate_id_dict()
        index = self.id_dict[im_id]
        return self.__getitem__(index)

    def get_coco_style_names(self,json_path, with_cat_id=False):
        json_path = os.path.expanduser(json_path)
        fpGt=COCO(json_path)
        #print(fpGt.cats.keys())
        if with_cat_id:
            names= {fpGt.cats[i]['id']:fpGt.cats[i]['name'] for i in fpGt.cats.keys()}
        else:
            names = [fpGt.cats[i]['name'] for i in fpGt.cats.keys()]
        return names

    def _load_image(self, img_path, backend='pillow', RGB=True):
        if backend == 'pillow':
            if RGB:
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.open(img_path).convert('BGR')
        elif backend == 'opencv':
            img = cv2.imread(img_path)
            if RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        return img