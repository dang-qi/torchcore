import torch
import inspect

from ...util.registry import Registry
from ...util.build import build_with_config

OPTIMIZER_REG = Registry('OPTIMIZER')
SCHEDULER_REG = Registry('SCHEDULER')

# from https://github.com/open-mmlab/mmcv/blob/f22c9eb4a409470b7e645f17fa1997fe85e27909/mmcv/runner/optimizer/builder.py
def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZER_REG.register(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers

def register_torch_schedulers():
    torch_schedulers = []
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue
        _scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler,
                                                  torch.optim.lr_scheduler._LRScheduler):
            SCHEDULER_REG.register(_scheduler)
            torch_schedulers.append(module_name)
    return torch_schedulers

TORCH_OPTIMIZERS = register_torch_optimizers()
TORCH_SCHEDULERS = register_torch_schedulers()

def build_optimizer(model, cfg ):
    if hasattr(model, 'module'):
            model = model.module
    cfg = cfg.copy()
    cfg.params = model.parameters()
    optimizer = build_with_config(cfg, OPTIMIZER_REG)
    return optimizer

def build_lr_scheduler(optimizer, cfg):
    cfg.optimizer=optimizer
    scheduler = build_with_config(cfg, SCHEDULER_REG)
    return scheduler