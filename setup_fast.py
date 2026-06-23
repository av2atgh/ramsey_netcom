"""Build the Cython generators:  python3 setup_fast.py build_ext --inplace"""
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "generators_fast",
    ["generators_fast.pyx"],
    language="c++",
    extra_compile_args=["-O3", "-std=c++11"],
)

setup(
    name="generators_fast",
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}),
)
