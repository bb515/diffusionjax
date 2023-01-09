sgm
===
sgm is a lightweight, open-source introduction to diffusion models, also known as score-based generative models (SGMs). It is implemented in Python via the autodiff framework, [JAX](https://github.com/google/jax). In particular, sgm uses the [Flax](https://github.com/google/flax) library for the neural network approximator of the score.

Based off the [Jupyter notebook](https://jakiw.com/sgm_intro) by Jakiw Pidstrigach, a tutorial on the theoretical and implementation aspects of SGMs.

The development of sgm has been supported by The Alan Turing Institute through the Theory and Methods Challenge Fortnights event “Accelerating generative models and nonconvex optimisation”, which took place on 6-10 June 2022 and 5-9 Sep 2022 at The Alan Turing Institute headquarters.

Does haves
-----------
- Training scores on (possibly, image) data and sampling from the generative model.
- Not many lines of code.
- Easy to use, extendable. Get started with the example, provided.

Doesn't haves
---------------
- Geometry other than Euclidean space, such as Riemannian manifolds.
- Diffusion in a latent space.
- Augmented with critically-damped Langevin diffusion.

Get started
------------

### Installation ###

- The package requires Python 3.9+.
- Clone the repository `git clone git@github.com:bb515/sgm.git`
- Install using pip `pip install -e .` from the root directory of the repository (see the `setup.py` for the requirements that this command installs).

### Running examples ###

- Run the example by typing `python examples/example.py` on the command line from the root directory of the repository.

