"""
Setup script for diffusionjax.

This setup is required or else
    >> ModuleNotFoundError: No module named 'diffusionjax'
will occur.
"""
from setuptools import setup, find_packages
import subprocess
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# The text of the LICENSE file
LICENSE = (HERE / "LICENSE.rst").read_text()

setup(
    name="diffusionjax",
    python_requires=">=3.8",
    description="diffusionjax is a simple and accessible diffusion models package in JAX",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/diffusionjax",
    author="Jakiw Pidstrigach and Benjamin Boys",
    license="MIT",
    license_file=LICENSE,
    packages=find_packages(exclude=['*.test']),
    include_package_data=True,
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
