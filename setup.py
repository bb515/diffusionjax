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
    version="0.1.0",
    description="A simple and accessible diffusion models package in JAX.",
    long_description=README,
    long_description_content_type="text/markdown",
    #url="",
    #author="Jakiw Pidstrigach and Benjamin Boys",
    #license="MIT",
    packages=find_packages(exclude=['*.test']),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'jaxlib>=0.4.1',
        'jax>=0.4.1',
        'optax',
        'flax',
        'backends>=1.4.32',
        'mlkernels>=0.3.6',
        ]
    )
