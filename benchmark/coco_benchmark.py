from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

class COCOresult(object):
    def __init__(self):
        self.result = []

    def update(self, batch_boxes, batch_class_ids, batch_scores, image_ids):
        for boxes, class_ids, scores, image_id in zip(batch_boxes, batch_class_ids, batch_scores, image_ids):
            for box, class_id, score in zip(boxes, class_ids, scores):
                result = {}
                result['bbox'] = box.tolist()
                result['image_id'] = image_id
                result['category_id'] = class_id.tolist()
                result['score'] = score.tolist()
                self.result.append(result)

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.result, f)
            print('The result is successfully saved in {}'.format(path))
            return
        print('Fail to save the result to {} !'.format(path))

def eval_result(dataset_name=None, groundtruth_path=None, target_path='temp_result.json'):
    '''just person result is generated in coco dataset'''

    if dataset_name is None and groundtruth_path is None:
        raise ValueError('Either dataset name or groundtruth path should be specified!')
    if not bool(dataset_name) and not bool(groundtruth_path):
        raise ValueError('Only dataset or groundtruth path should be specified')
    if groundtruth_path:
        gt_json = groundtruth_path
    else:
        if dataset_name not in ['coco', 'modanet']:
            raise ValueError('only support coco and modanet dataset')
        if dataset_name == 'coco':
            gt_json='/ssd/data/datasets/COCO/annotations/instances_val2014.json'
        elif dataset_name == 'modanet':
            gt_json='/ssd/data/datasets/modanet/Annots/modanet_instances_val.json'
    annType = 'bbox'
    cocoGt=COCO(gt_json)
    cocoDt=cocoGt.loadRes(target_path)

    imgIds=sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    if dataset_name == 'coco':
        cocoEval.params.catIds = [1]
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()