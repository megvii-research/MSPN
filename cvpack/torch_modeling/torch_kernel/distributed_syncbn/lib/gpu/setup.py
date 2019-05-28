from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='syncbn_gpu',
      ext_modules=[CUDAExtension('syncbn_gpu', ['syncbn_cuda.cpp', 'syncbn_cuda_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})