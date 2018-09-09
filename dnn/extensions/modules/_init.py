import os
import sys
import platform

import torch

if platform.system() == 'Darwin' :
    filepath = os.path.realpath(__file__)
    dirpath = os.path.dirname(filepath)

    mac_ver = '.'.join( platform.mac_ver()[0].split('.')[:2] )
    python_ver = '.'.join( [ str(sys.version_info.major), str(sys.version_info.minor) ] )
    machine = platform.machine()
    package_name = 'lib.macosx-%s-%s-%s' % ( mac_ver, machine, python_ver )

    libpath = os.path.join( dirpath, '../build', package_name )

    if libpath not in sys.path :
        sys.path.append( libpath )
elif platform.system() == "Linux" :
    filepath = os.path.realpath(__file__)
    dirpath = os.path.dirname(filepath)

    python_ver = '.'.join( [ str(sys.version_info.major), str(sys.version_info.minor) ] )
    machine = platform.machine()
    package_name = 'lib.linux-%s-%s' % ( machine, python_ver )
    libpath = os.path.join( dirpath, '../build', package_name )
    if libpath not in sys.path :
        sys.path.append( libpath )
else :
    raise NotImplementedError
