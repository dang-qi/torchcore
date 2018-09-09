import os
import torch

def get_device( gpu ):
    if gpu is not None :
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    use_cuda = torch.cuda.is_available()
    device_name = "cuda" if use_cuda else "cpu"
    if use_cuda and torch.cuda.device_count() > 1 and gpu is not None :
        device_name = device_name + ':' + gpu
    return torch.device( device_name )
