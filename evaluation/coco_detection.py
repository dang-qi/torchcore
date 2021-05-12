import os
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
class COCOEvaluator():
    def __init__(self, evaluate_type='bbox', dataset_name=None, gt_path=None) -> None:
        assert dataset_name is not None or gt_path is not None
        self.dataset_name = dataset_name
        if gt_path is not None:
            self.gt_path = gt_path
        else:
            self.set_gt_path(dataset_name)
        self.evaluate_type = evaluate_type

    def set_gt_path(self, dataset_name):
        if dataset_name not in ['coco','coco_person', 'modanet', 'fashionpedia']:
            raise ValueError('only support coco, coco_person and modanet dataset')
        if dataset_name == 'coco_person':
            gt_json=os.path.expanduser('~/data/datasets/COCO/annotations/instances_val2014.json')
        elif dataset_name == 'coco':
            gt_json=os.path.expanduser('~/data/datasets/COCO/annotations/instances_val2017.json')
        elif dataset_name == 'fashionpedia':
            gt_json=os.path.expanduser('~/data/datasets/Fashionpedia/annotations/instances_attributes_val2020.json')
        elif dataset_name == 'modanet':
            gt_json=os.path.expanduser('~/data/datasets/modanet/Annots/modanet_instances_val.json')

        self.gt_path = gt_json

    def evaluate(self, result_path):
        if isinstance(self.evaluate_type, str):
            self.evaluate_once(result_path, self.evaluate_type)
        else:
            for eval_type in self.evaluate_type:
                assert eval_type in ['segm','bbox','keypoints']
                self.evaluate_once(result_path, eval_type)

    def evaluate_once(self, result_path, eval_type):
        # only support bbox mode for now
        dataset = self.dataset_name
        # we need to map the category ids back
        if dataset == 'coco':
            print('revise coco dataset label to gt')
            cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
            with open(result_path) as f:
                results = json.load(f)
                for result in results:
                    temp_id = result['category_id']
                    result['category_id'] = cat_ids[temp_id-1]
            result_path = 'coco_'+result_path
            with open(result_path,'w') as f:
                json.dump(results,f)
        if dataset == 'fashionpedia':
            print('revise fashionpedia dataset label to gt')
            with open(result_path) as f:
                results = json.load(f)
                for result in results:
                    temp_id = result['category_id']
                    result['category_id'] = temp_id-1
            result_path = 'fashionpedia_'+result_path
            with open(result_path,'w') as f:
                json.dump(results,f)
        annType = eval_type
        cocoGt=COCO(self.gt_path)
        cocoDt=cocoGt.loadRes(result_path)

        imgIds=sorted(cocoGt.getImgIds())

        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        if dataset == 'coco_person':
            cocoEval.params.catIds = [1]
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()