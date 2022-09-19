"""
Setup script for sgm.

This setup is required or else
    >> ModuleNotFoundError: No module named 'sgm'
will occur.
"""
from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name="sgm",
    #version="0.1.0",
    #description="",
    long_description=README,
    long_description_content_type="text/markdown",
    #url="",
    #author="Benjamin Boys",
    #license="MIT",
    packages=find_packages(exclude=['*.test']),
    install_requires=[
        'backends',
        'mlkernels',
        'numpy',
        'scipy',
        'tqdm',
        'h5py',
        'matplotlib',
        ]
    )
