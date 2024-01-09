"""
Setup script for diffusionjax.

This setup is required or else
    >> ModuleNotFoundError: No module named 'diffusionjax'
will occur.
"""
from setuptools import setup, find_packages
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# The text of the LICENSE file
LICENSE = (HERE / "LICENSE.rst").read_text()

setup(
  name="diffusionjax",
  # python_requires=">=3.8",
  description="diffusionjax is a simple and accessible diffusion models package in JAX",
  long_description=README,
  long_description_content_type="text/markdown",
  url="https://github.com/bb515/diffusionjax",
  author="Benjamin Boys and Jakiw Pidstrigach",
  license="MIT",
  license_file=LICENSE,
  packages=find_packages(exclude=["*.test"]),
  install_requires=[
    "numpy",
    "scipy",
    "matplotlib",
    "flax",
    "ml_collections",
    "tqdm",
    "absl-py",
    "wandb",
    ],
  extras_require={
    'linting': [
      "flake8",
      "pylint",
      "mypy",
      "typing-extensions",
      "pre-commit",
      "ruff",
      'jaxtyping',
    ],
    'testing': [
      "optax",
      "orbax-checkpoint",
      "torch",
      "pytest",
      "pytest-xdist",
      "pytest-cov",
      "coveralls",
      "jax>=0.4.1",
      "jaxlib>=0.4.1",
      "setuptools_scm[toml]",
      "setuptools_scm_git_archive",
    ],
    'examples': [
      "optax",
      "orbax-checkpoint",
      "torch",
      "mlkernels",
    ],
  },
  include_package_data=True)
