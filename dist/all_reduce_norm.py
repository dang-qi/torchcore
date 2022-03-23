from pyparsing import Or
import torch
import torch.distributed as dist
import pickle
from .dist import get_world_size, _get_global_gloo_group
from collections import OrderedDict
from ..dnn.networks.base import norm

def py_obj_to_tensor(py_obj, device='cuda'):
    py_obj = pickle.dumps(py_obj)
    storage = torch.ByteStorage.from_buffer(py_obj)
    return torch.ByteTensor(storage).to(device=device)


def tensor_to_py_obj(tensor):
    return pickle.loads(tensor.cpu().numpy().tobytes())

def get_norm_states(module):
    NORM_LAYERS = tuple(norm.NORM_LAYERS.values())
    norm_states = OrderedDict()
    for name,m in module.named_modules():
        if isinstance(m, NORM_LAYERS):
            for k,v in m.state_dict().items():
                norm_states['{}.{}'.format(name,k)] = v
    return norm_states

def all_reduce(state_dict, op='sum', group=None):
    assert op in ['sum', 'mean']
    world_size = get_world_size()
    if world_size == 1:
        return state_dict
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return state_dict

    state_dict_keys = list(state_dict.keys())
    state_dict_keys = py_obj_to_tensor(state_dict_keys)
    dist.broadcast(state_dict_keys, src=0, group=group)
    state_dict_keys = tensor_to_py_obj(state_dict_keys)

    v_shapes = [s.shape for s in state_dict.values()]
    v_size = [s.numel() for s in state_dict.values()]
    v_flat = [s.flatten() for s in state_dict.values()]
    cat_v_flat = torch.cat(v_flat)

    dist.all_reduce(cat_v_flat, op=dist.ReduceOp.SUM, group=group)
    if op == 'mean':
        cat_v_flat = cat_v_flat / world_size
    
    split_v = torch.split(cat_v_flat, v_size)
    return OrderedDict({k:v.reshape(v_s) for k, v, v_s in zip(state_dict_keys, split_v, v_shapes)})
    



def all_reduce_norm(module, group=None):
    '''reduce all the normalization parameters from model'''
    norm_states = get_norm_states(module)
    norm_states = all_reduce(norm_states, op='mean', group=group)
    module.load_state_dict(norm_states, strict=False)