import torch
import inspect
from ...util.registry import Registry
from ...util.build import build_with_config
from .distributed_sampler_wrapper import DistributedSamplerWrapper

SAMPLER_REG = Registry('DATASET SAMPLER')

def register_torch_sampler():
    torch_sampler = []
    for module_name in dir(torch.utils.data.sampler):
        if module_name.startswith('__'):
            continue
        _sampler = getattr(torch.utils.data.sampler, module_name)
        if inspect.isclass(_sampler) and issubclass(_sampler, torch.utils.data.sampler.Sampler):
            SAMPLER_REG.register(_sampler)
            torch_sampler.append(module_name)
    return torch_sampler

TORCH_SAMPLER = register_torch_sampler()


def build_sampler(cfg, distributed=False, grouped=True):
    sampler = build_with_config(cfg, SAMPLER_REG)
    if distributed:
        sampler = DistributedSamplerWrapper(sampler)
    return sampler
