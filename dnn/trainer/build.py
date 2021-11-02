import torch
import inspect

from ...util.registry import Registry
from ...util.build import build_with_config

TRAINER_REG = Registry('Trainer')

def build_trainer(cfg, default_args=None):
    trainer = build_with_config(cfg, TRAINER_REG, default_args=default_args)
    return trainer
