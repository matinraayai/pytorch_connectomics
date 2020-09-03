import os
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, find_packages


def get_include():
    dir_name = get_python_inc()
    return [dir_name, os.path.dirname(dir_name), np.get_include()]


def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/zudi-lin/pytorch_connectomics'

    setup(name='connectomics',
          description='Automatic Reconstruction of Connectomics with PyTorch',
          version=__version__,
          url=url,
          license='MIT',
          author='Zudi Lin, Donglai Wei',
          install_requires=['scipy'],
          include_dirs=get_include(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
