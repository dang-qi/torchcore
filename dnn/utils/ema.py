from this import d
import torch
import copy
import math
from torch import nn
def is_parallal_model(model):
    return isinstance(model,(nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

class ModelEMA():
    '''Exponetial Moving Average model'''
    def __init__(self, model, decay=0.9999, num_updates=None):
        self.ema = copy.deepcopy(model.module if is_parallal_model(model) else model)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        if num_updates is None:
            self.num_updates=0
        else:
            self.num_updates=num_updates

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.num_updates += 1
            d = self.decay(self.num_updates)

            model_state_dict = model.module.state_dict() if is_parallal_model(model) else model.state_dict()

            for k,v in self.ema.state_dict():
                # the non-floating point type is bn.num_batches_tracked
                if v.dtype.is_floating_point: 
                    v *=d
                    v += (1-d)*model_state_dict[k].detach()