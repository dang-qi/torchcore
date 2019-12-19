from .dataset import dataset
from .dataset_new import Dataset
from .coco_dataset import coco_dataset
from .cocowider_dataset import cocowider_dataset
from .deepfashion_dataset import deepfashion_dataset
from .deepfashion2_dataset import deepfashion2_dataset
from .deepfashionp_dataset import deepfashionp_dataset
from .figuratic_dataset import figuratic_dataset
from .aflw_dataset import aflw_dataset
from .aflw_patch_dataset import aflw_patch_dataset
from .coco_person_patch_dataset import coco_person_patch_dataset
from .modanet_dataset import modanet_dataset
from .modanetp_dataset import modanetp_dataset
from .fire_dataset import fire_dataset
from .coco_person import COCOPersonDataset
from .modanet_dataset_new import ModanetDataset
from .modanet_human_dataset import ModanetHumanDataset

selector = {}
selector['coco'] = coco_dataset
selector['fire'] = fire_dataset
selector['cocowider'] = cocowider_dataset
selector['deepfashion'] = deepfashion_dataset
selector['deepfashion2'] = deepfashion2_dataset
selector['deepfashionp'] = deepfashionp_dataset
selector['figuratic'] = figuratic_dataset
selector['aflw'] = aflw_dataset
selector['aflw_patch'] = aflw_patch_dataset
selector['coco_person_patch'] = coco_person_patch_dataset
selector['modanet'] = modanet_dataset
selector['modanetp'] = modanetp_dataset

def load_dataset( cfg, params, settings=None, **kwargs ):
    dataset = selector[ params[0] ]( cfg.dataset, *( params[1:] ) )
    if settings is None :
        settings = cfg.dataset.WORKSET_SETTINGS
    dataset.load(settings,**kwargs)
    return dataset
