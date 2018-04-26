"""Python Makefile for cython
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(name='csptools', ext_modules=cythonize("csptools.pyx"))
