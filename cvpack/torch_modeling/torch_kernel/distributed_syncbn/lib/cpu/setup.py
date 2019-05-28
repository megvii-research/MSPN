from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='syncbn_cpu',
      ext_modules=[CppExtension('syncbn_cpu', ['syncbn_cpu.cpp'])],
      cmdclass={'build_ext': BuildExtension})