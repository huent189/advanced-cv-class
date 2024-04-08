from setuptools import Extension, setup
from Cython.Build import cythonize
import sys
import numpy
if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'


ext_modules = [
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='parallel-tutorial',
    ext_modules=cythonize(ext_modules),
)