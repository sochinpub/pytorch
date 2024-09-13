from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='myAdd',
    packages=find_packages(),
    version='0.1.0',
    author='muzhan',
    ext_modules=[
        CUDAExtension(
            'sum_double',
            [
                './add/add.cpp',
                './add/add_cuda.cu'
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)