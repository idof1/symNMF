from setuptools import setup, Extension
import numpy

module = Extension('symnmfmodule', sources=['symnmfmodule.c','symnmf.c'], include_dirs=[numpy.get_include()])

setup(
    name='symnmfmodule',
    version='1.0',
    description='SymNMF Algorithm Module',
    ext_modules=[module],
)
