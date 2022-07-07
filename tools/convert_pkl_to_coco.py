import os
import pickle
import json
from datetime import date

def convert_pkl_to_coco_json(example_coco, pkl_path, json_path, part, info, categories, licenses=None):
    '''
        Convert pkl annotation to coco json file
        paramenters:
            example_coco: path of example coco json file
            info: dict. One example could be: 
                    {'description': 'COCO 2014 Dataset',
                    'url': 'http://cocodataset.org',
                    'version': '1.0',
                    'year': 2014,
                    'contributor': 'COCO Consortium',
                    'date_created': '2017/09/01'}
            categories: list of dict. For example:
                        [{'supercategory': 'person', 'id': 1, 'name': 'person'},
                        {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                        {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}]
            images: list of dict, the info of images. For example:
                        [{'license': 3,
                        'file_name': 'COCO_val2014_000000391895.jpg',
                        'coco_url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg',
                        'height': 360,
                        'width': 640,
                        'date_captured': '2013-11-14 11:18:45',
                        'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
                        'id': 391895}]
    '''
    example_coco = os.path.expanduser(example_coco)
    assert os.path.exists(example_coco) 
    with open(example_coco, 'r') as f:
        data = json.load(f)
    with open(pkl_path, 'rb') as f:
        anno = pickle.load(f)[part]
    data['info'] = info
    data['categories'] = categories
    images = []
    annotations = []
    for im in anno:
        single_im = {}
        single_im['file_name'] = im['file_name']
        single_im['height'] = im['height']
        single_im['width'] = im['width']
        single_im['id'] = im['id']
        images.append(single_im)
        annotations.append(im['objects'])
    data['images'] = images
    data['annotations'] = annotations
    if licenses is not None:
        data['licenses'] = licenses

    with open(json_path,'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    coco_path = os.path.expanduser('~/data/datasets/COCO/annotations/instances_val2014.json')
    parts = ['train', 'test', 'extra']
    for p in parts:
        pkl_path = 'data/{}.pkl'.format(p)
        json_path = 'data/SVHN_{}.json'.format(p)
        today = date.today()
        # dd/mm/YY
        d1 = today.strftime("%Y/%m/%d")
        info = {'description': 'SVHN Dataset',
                    'url': 'http://ufldl.stanford.edu/housenumbers/',
                    'version': '1.0',
                    'year': 2011,
                    'contributor': 'Unknow',
                    'date_created': d1}
        categories = [{'supercategory':'house number', 'id': i, 'name': '{}'.format(i%10)} for i in range(1,11)]
        convert_pkl_to_coco_json(coco_path, pkl_path, json_path, p, info, categories)