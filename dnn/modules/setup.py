import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

def get_modules():
    modules = []
    modules.append( CppExtension('roi_pool_cpu', ['roi_pool_cpu.cpp']))
    modules.append( CppExtension('roi_align_cpu', ['roi_align_cpu.cpp']))
    modules.append( CppExtension('nms_cpu', ['nms_cpu.cpp']))
    if torch.cuda.is_available() :
        modules.append(CUDAExtension('roi_pool_cuda', ['roi_pool_cuda.cpp','roi_pool_kernel.cu']))
        modules.append(CUDAExtension('roi_align_cuda', ['roi_align_cuda.cpp','roi_align_kernel.cu']))
    return modules


setup(
    name='torchcore',
    ext_modules=get_modules(),
    cmdclass={
        'build_ext': BuildExtension
    })
