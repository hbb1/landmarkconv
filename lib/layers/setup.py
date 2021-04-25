# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CppExtension

import os
import torch
import shutil
from os import path
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension
import pdb
import glob
torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 0], "Requires PyTorch >= 1.0"

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []
    # if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        sources += source_cuda
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
                '-DCUDA_HOST_COMPILER=/usr/bin/gcc5',
                # '-D__CUDA_NO_HALF_OPERATORS__',
                # '-D__CUDA_NO_HALF_CONVERSIONS__',
                # '-D__CUDA_NO_HALF2_OPERATORS__',
                # '-DCUDA_HOST_COMPILER=/usr/bin/gcc5',
                '-gencode', 'arch=compute_30,code=sm_30',
                '-gencode', 'arch=compute_35,code=sm_35',
                '-gencode', 'arch=compute_37,code=sm_37',
                '-gencode', 'arch=compute_50,code=sm_50',
                '-gencode', 'arch=compute_52,code=sm_52',
                '-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61',
                '-gencode', 'arch=compute_70,code=sm_70',
            ]
        }
    else:
        raise EnvironmentError('CUDA is required to compile MMDetection!')
    
    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))
    

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "landmarkconv._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="landmarkconv",
    ext_modules= get_extensions(),
    cmdclass={
        "build_ext": BuildExtension
    }
)