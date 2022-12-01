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
    #description="",
    long_description=README,
    long_description_content_type="text/markdown",
    #url="",
    #author="Jakiw Pidstrigach and Benjamin Boys",
    #license="MIT",
    packages=find_packages(exclude=['*.test']),
    install_requires=[
        'numpy==1.23.4',
        'scipy==1.9.2',
        'tqdm==4.64.1',
        'matplotlib==3.6.1',
        'optax==0.1.3',
        'jax==0.3.23',
        'jaxlib==0.3.22',
        'flax==0.6.1'
        ]
    )
