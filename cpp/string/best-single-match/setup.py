# setup.py
from distutils.core import setup, Extension

module1 = Extension("strutil", sources=["main.cpp"])

setup(name="strutil", version="0.0.1", description="strutil", ext_modules=[module1])
