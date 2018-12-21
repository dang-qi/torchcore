import os
import torch
import platform
from distutils.core import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

version='0.0.1'

dirname = os.path.dirname( os.path.realpath(__file__) )

def cpu_module( module_name ):
    module = CppExtension( module_name,['%s.cpp' % (module_name)] )
    module.include_dirs.append( os.path.join(dirname, "modules_cpu") )
    if platform.system() == "Linux" :
        module.extra_objects.append(os.path.join(dirname, 'build/modules_cpu/libmodules_cpu.so'))
    elif platform.system() == "Darwin" :
        module.library_dirs.append( os.path.join(dirname, 'build/modules_cpu'))
        module.libraries.append('modules_cpu')
    else :
        raise NotImplementedError
    return module

def gpu_module( module_name ):
    module = CppExtension( module_name,['%s.cpp' % (module_name)] )
    module.include_dirs.append( os.path.join(dirname, "modules_gpu") )
    if platform.system() == "Linux" :
        module.extra_objects.append(os.path.join(dirname, 'build/modules_gpu/libmodules_gpu.so'))
    else :
        raise NotImplementedError
    return module

setup(
        name='overlaps_cpu',
        version=version,
        ext_modules=[ cpu_module('overlaps_cpu') ],
        cmdclass={'build_ext': BuildExtension}
)

setup(
        name='roipool_cpu',
        version=version,
        ext_modules=[ cpu_module('roipool_cpu') ],
        cmdclass={'build_ext': BuildExtension}
)

setup(
        name='crop_and_resize_cpu',
        version=version,
        ext_modules=[ cpu_module('crop_and_resize_cpu') ],
        cmdclass={'build_ext': BuildExtension}
)

setup(
        name='nms_cpu',
        version=version,
        ext_modules=[ cpu_module('nms_cpu') ],
        cmdclass={'build_ext': BuildExtension}
)

if torch.cuda.is_available() :
    setup(
        name='overlaps_gpu',
        version=version,
        ext_modules=[ gpu_module('overlaps_gpu') ],
        cmdclass={'build_ext': BuildExtension}
    )

    setup(
            name='roipool_gpu',
            version=version,
            ext_modules=[ cpu_module('roipool_gpu') ],
            cmdclass={'build_ext': BuildExtension}
    )

    setup(
        name='crop_and_resize_gpu',
        version=version,
        ext_modules=[ gpu_module('crop_and_resize_gpu') ],
        cmdclass={'build_ext': BuildExtension}
    )
