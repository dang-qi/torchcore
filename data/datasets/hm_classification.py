import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from .dataset_new import Dataset
from ..transforms import Compose
from ..transforms.build import build_transform
from .build import DATASET_REG

@DATASET_REG.register(force=True)
class HMClassificationDataset(Dataset):
    '''

    article keys: 
       ['article_id', 'product_code', 'prod_name', 'product_type_no',
       'product_type_name', 'product_group_name', 'graphical_appearance_no',
       'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
       'perceived_colour_value_id', 'perceived_colour_value_name',
       'perceived_colour_master_id', 'perceived_colour_master_name',
       'department_no', 'department_name', 'index_code', 'index_name',
       'index_group_no', 'index_group_name', 'section_no', 'section_name',
       'garment_group_no', 'garment_group_name', 'detail_desc', 'path']
    '''
    def __init__(self, root_path, transforms=None):
        self.root_path = os.path.expanduser(root_path)
        self._root = self.root_path
        self.article_path = os.path.join(root_path, 'articles.csv')
        #self.customer_path = os.path.join(root_path, 'customers.csv')
        #self.transaction_path = os.path.join(root_path, 'transactions_train.csv')
        if transforms is None:
            transforms = []
        self._transforms=Compose(transforms)

        self.articles = pd.read_csv(self.article_path, dtype={'article_id':str})
        #print(self.articles)
        self.add_image_path()

        # remove the ones that doesn't has image
        self.remove_empty_path()

        # remove the ones that has 'Unknow' lable
        self.remove_unknow()
        # merge 'Umbrella'
        self.merge_umbrella()

        self.convert_to_image_format()

    def get_categories(self):
        name_dict = dict()
        for num,name in zip(self.articles['product_type_no'], self.articles['product_type_name']):
            if num not in name_dict:
                name_dict[num]=name
            else:
                if(name_dict[num]!=name):
                    print('wrong', num,name)
        return name_dict

    # there is a extract by id func
    #def get_image_by_id(self, im_id):
    #    if not hasattr(self, 'id_index_dict'):
    #        self.id_index_dict = {}
    #        for i,im in enumerate(self._images):
    #            the_id = im['id']
    #            self.id_index_dict[the_id] = i
    #    return self.__getitem__(self.id_index_dict[im_id])


    def convert_to_image_format(self):
        self._images = []
        article_num = len(self.articles)
        for i in range(article_num):
            image = {}
            for k in self.articles.keys():
                if k == 'article_id':
                    image['id'] = self.articles['article_id'].iat[i]
                elif k == 'product_type_name':
                    image['label_text'] = self.articles['product_type_name'].iat[i]
                elif k == 'product_type_no':
                    image['label_num'] = int(self.articles['product_type_no'].iat[i])
                elif k == 'detail_desc':
                    image['description'] = self.articles['detail_desc'].iat[i]
                    if not isinstance(image['description'],str):
                        image['description'] = ""
                else:
                    image[k] = self.articles[k].iat[i]
            self._images.append(image)
                    


    #def convert_to_image_format(self):
    #    self._images = []
    #    article_num = len(self.articles)
    #    for i in range(article_num):
    #        image = {}
    #        image['path'] = self.articles['path'].iat[i]
    #        image['id'] = self.articles['article_id'].iat[i]
    #        image['label_text'] = self.articles['product_type_name'].iat[i]
    #        if not isinstance(image['label_text'],str):
    #            print('wrong',image['label_text'])
    #        image['label_num'] = int(self.articles['product_type_no'].iat[i])
    #        image['description'] = self.articles['detail_desc'].iat[i]
    #        if not isinstance(image['description'],str):
    #            image['description'] = ""

    #        self._images.append(image)

    def map_category_id_to_continous(self,start_id=1):
        id_set = set()
        for image in self._images:
            id_set.add(image['label_num'])
        ids = sorted(list(id_set))
        self.id_map = {aid:i+start_id for i, aid in enumerate(ids)}
        self.cast_id_map = {i+start_id:aid for i, aid in enumerate(ids)}
        #print(id_map)
        for image in self._images:
            image['label_num'] = self.id_map[image['label_num']]
        
    def remove_empty_path(self):
        self.articles = self.articles[self.articles['path'].apply(os.path.exists)]

    def remove_unknow(self):
        self.articles = self.articles[self.articles['product_type_name']!='Unknown']

    def merge_umbrella(self):
        for i in range(len(self.articles)):
            if self.articles['product_type_name'].iat[i]=='Umbrella':
                #self.articles.iat['product_type_name',i] = 83
                self.articles['product_type_no'].iat[i] = 83

    def add_image_path(self):
        im_path = []
        base_path = 'images'
        for _,a in self.articles.iterrows():
            folder = a['article_id'][:3]
            file_name = a['article_id'] + '.jpg'
            path = os.path.join(self.root_path,base_path, folder, file_name)
            im_path.append(path)
        self.articles['path'] = im_path

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        path = image['path']
        label_text = image['label_text']
        description = image['description']
        #path = self.articles['path'].iat[idx]
        #article_id = self.articles['article_id']
        article_type = self.articles['product_type_no'].iat[idx]
        img = self._load_image(path)
        inputs = {}
        targets = {}
        inputs['data'] = img
        if self._transforms is not None:
            inputs, targets = self._transforms(inputs, targets)
        inputs['label_text'] = label_text
        inputs['description'] = description
        inputs['label_num'] = image['label_num']
        inputs['id'] = image['id']
        #inputs['article_id'] = article_id
        #targets['label'] = 
        #if self._transforms is not None:
        #    for t in self._transforms:
        #        inputs, targets = t(inputs, targets)
        
        return inputs, targets

    def show_example_by_index(self, ind=None,show_image=True):
        '''
            Show random example of the dataset
            if ind is None, then show random sample
        '''
        if ind is None:
            ind = random.randint(0,len(self)-1)
        assert ind >=0 and ind <len(self)
        
        image = self._images[ind]
        if show_image:
            im = self._load_image(image['path'])
            plt.figure()
            plt.imshow(im)
        pprint.pprint(self._images[ind])

    def default_include_keys(self):
        include_keys = ['prod_name',
            'product_type_name', 'product_group_name', 
            'graphical_appearance_name', 'colour_group_name',
            'perceived_colour_value_name',
            'perceived_colour_master_name',
            'department_name','index_name',
            'index_group_name', 'section_name',
            'garment_group_name']
        return include_keys


    def get_dataset_value_range(self, include_keys=None):
        if include_keys is None:
            include_keys = self.default_include_keys()
        
        result = {}
        for k in include_keys:
            result[k] = set()
        for k in self.articles.keys():
            if k in include_keys:
                for i in range(len(self.articles[k])):
                    result[k].add(self.articles[k].iat[i])
        return result

    def generate_attribute_dict(self, include_keys=None):
        print('Generate attribute dict...')
        if include_keys is None:
            include_keys = self.default_include_keys()
        k_dict = self.get_dataset_value_range(include_keys)
        attribute_dict = {}
        for k in k_dict:
            attribute_dict[k] = {v:[] for v in k_dict[k]}
        for k in self.articles.keys():
            if k in include_keys:
                for i in range(len(self.articles[k])):
                    attri = self.articles[k].iat[i]
                    attribute_dict[k][attri].append(i)
        self.attribute_dict = attribute_dict
        print('Attribute dict generation done!')
        

    def show_dataset_by_attribute(self, name, value, num=1, include_keys=None, force_regenerate=False):
        if not hasattr(self, 'attribute_dict') or force_regenerate:
            self.generate_attribute_dict(include_keys=include_keys)
        ind_list = self.attribute_dict[name][value]
        sample_num = min(num,len(ind_list))
        sample = random.sample(ind_list, sample_num)
        print(f'There are {len(ind_list)} samples in {name} {value}')
        for ind in sample:
            self.show_example_by_index(ind)
                

    def merge_list(self):
        merge_list = {'product_group_name':[('Items','Accessories'),('')]}
        remove_list = {'product_group_name':['Interior textile','Fun','Stationery','Garment and Shoe care','Cosmetic','Furniture'],
                       'graphical_appearance_name':['All over pattern', 'Other pattern','Other structure','Unknown']}

