from torch.utils.data import DataLoader, sampler
from ...util.registry import Registry
from ...util.build import build_with_config
from ..sampler.build import build_sampler
from ..collate.build import build_collate

DATASET_REG = Registry('DATASET')

def build_dataset(cfg):
    dataset = build_with_config(cfg, DATASET_REG)
    return dataset

def build_dataloader(cfg):
    cfg = cfg.copy()
    dataset_cfg = cfg.pop('dataset')
    dataset = build_dataset(dataset_cfg)

    sampler_cfg = cfg.pop('sampler', None)
    if sampler_cfg is not None:
        sampler = build_sampler(sampler_cfg)
    else:
        sampler = None

    collate_cfg = cfg.pop('collate', None)
    if collate_cfg is not None:
        collate = build_collate(collate_cfg)
    else:
        collate = None

    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=collate, **cfg)
    return dataloader