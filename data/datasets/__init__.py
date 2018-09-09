from .dataset import dataset
from .coco_dataset import coco_dataset
from .cocowider_dataset import cocowider_dataset
from .deepfashion_dataset import deepfashion_dataset
from .aflw_dataset import aflw_dataset

selector = {}
selector['coco'] = coco_dataset
selector['cocowider'] = cocowider_dataset
selector['deepfashion'] = deepfashion_dataset
selector['aflw'] = aflw_dataset
