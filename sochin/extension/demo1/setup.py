from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='ncrelu_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            'ncrelu_cpp', ['ncrelu.cpp']
        )
    ],
    cmdclass= {
        'build_ext' : cpp_extension.BuildExtension
    }
)