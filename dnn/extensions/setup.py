import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

modules = []

# ROIPool Modules
modules.append( CppExtension('roi_pool_cpu', ['roi_pool_binding.cpp']) )
if torch.cuda.is_available() :
    modules.append( CUDAExtension('roi_pool_cuda', ['roi_pool_cuda.cpp','roi_pool_kernel.cu',]) )


setup(
    name='roi_pool_cpp',
    ext_modules=modules
    ,
    cmdclass={
        'build_ext': BuildExtension
    })
