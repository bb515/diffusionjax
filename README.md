diffusionjax
============

A simple and accessible diffusion models package in JAX.

diffusionjax is a simple, accessible introduction to diffusion models, also known as score-based generative models (SGMs). It is implemented in Python via the autodiff framework, [JAX](https://github.com/google/jax). In particular, sgm uses the [Flax](https://github.com/google/flax) library for the neural network approximator of the score.

Based off the [Jupyter notebook](https://jakiw.com/sgm_intro) by Jakiw Pidstrigach, a tutorial on the theoretical and implementation aspects of SGMs.

The development of sgm has been supported by The Alan Turing Institute through the Theory and Methods Challenge Fortnights event "Accelerating generative models and nonconvex optimisation", which took place on 6-10 June 2022 and 5-9 Sep 2022 at The Alan Turing Institute headquarters.

Contents:

- [Installation](#installation)
- [Examples](#examples)
    - [Introduction to diffusion models](#introduction-to-diffusion-models)
- [Does haves](#does-haves)
- [Doesn't haves](#doesn't-haves)
- [References](#references)

## Installation
The package requires Python 3.9+. `pip install sgm` or for developers,
- Clone the repository `git clone git@github.com:bb515/sgm.git`
- Install using pip `pip install -e .` from the root directory of the repository (see the `setup.py` for the requirements that this command installs).

## Examples

### Introduction to diffusion models
- Run the example by typing `python examples/example.py` on the command line from the root directory of the repository.
```python
>>> num_epochs = 4000
>>> rng = random.PRNGKey(2023)
>>> rng, step_rng = random.split(rng, 2)
>>> num_samples = 8
>>> samples = sample_circle(num_samples)
>>> N = samples.shape[1]
>>> plot_samples(samples=samples, index=(0, 1), fname="samples", lims=((-3, 3), (-3, 3)))
```
![Prediction](readme_samples.png)
```python
# Get sde model
>>> sde = OU()
>>>
>>> def log_hat_pt(x, t):
>>>     """
>>>     Empirical distribution score.
>>>
>>>     Args:
>>>     x: One location in $\mathbb{R}^2$
>>>     t: time
>>>     Returns:
>>>     The empirical log density, as described in the Jupyter notebook
>>>     .. math::
>>>         \hat{p}_{t}(x)
>>>     """
>>>     mean, std = sde.marginal_prob(samples, t)
>>>     potentials = jnp.sum(-(x - mean)**2 / (2 * std**2), axis=1)
>>>     return logsumexp(potentials, axis=0, b=1/num_samples)
>>>
>>> # Get a jax grad function, which can be batched with vmap
>>> nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0), out_axes=(0)))
>>>
>>> # Running the reverse SDE with the empirical drift
>>> plot_score(score=nabla_log_hat_pt, t=0.01, area_min=-3, area_max=3, fname="empirical score")
```
![Prediction](readme_empirical_score.png)
```python
>>> sampler = EulerMaruyama(sde, nabla_log_hat_pt).get_sampler()
>>> q_samples = sampler(rng, n_samples=5000, shape=(N,))
>>> plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap empirical score")
```
![Prediction](readme_heatmap_empirical_score.png)
```python
>>> # What happens when I perturb the score with a constant?
>>> perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 1
>>> rng, step_rng = random.split(rng)
>>> sampler = EulerMaruyama(sde, perturbed_score).get_sampler()
>>> q_samples = sampler(rng, n_samples=5000, shape=(N,))
>>> plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap bounded perturbation")
```
![Prediction](readme_heatmap_bounded_perturbation.png)
```python
>>> # Neural network training via score matching
>>> batch_size=16
>>> score_model = MLP()
>>> # Initialize parameters
>>> params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
>>> # Initialize optimizer
>>> opt_state = optimizer.init(params)
>>> # Get loss function
>>> loss = get_loss_fn(
>>>     sde, score_model, score_scaling=True, likelihood_weighting=False,
>>>     reduce_mean=True, pointwise_t=False)
>>> # Train with score matching
>>> score_model, params, opt_state, mean_losses = retrain_nn(
>>>     update_step=update_step,
>>>     num_epochs=num_epochs,
>>>     step_rng=step_rng,
>>>     samples=samples,
>>>     score_model=score_model,
>>>     params=params,
>>>     opt_state=opt_state,
>>>     loss_fn=loss,
>>>     batch_size=batch_size)
>>> # Get trained score
>>> trained_score = get_score_fn(sde, score_model, params, score_scaling=True)
>>> plot_score(score=trained_score, t=0.01, area_min=-3, area_max=3, fname="trained score")
```
![Prediction](readme_heatmap_trained_score.png)
```python
>>> sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
>>> q_samples = sampler(rng, n_samples=1000, shape=(N,))
>>> plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap trained score")
```
![Prediction](readme_trained_score.png)

## Does haves
- Training scores on (possibly, image) data and sampling from the generative model.
- Not many lines of code.
- Easy to use, extendable. Get started with the example, provided.

## Doesn't haves
- Geometry other than Euclidean space, such as Riemannian manifolds.
- Diffusion in a latent space.
- Augmented with critically-damped Langevin diffusion.

## References
Algorithms in this package were ported from pre-existing code. In particular, the code was ported from the following papers and repositories:

The [official implementation](https://github.com/yang-song/score_sde) for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS) by [Yang Song](https://yang-song.github.io), [Jascha Sohl-Dickstein](http://www.sohldickstein.com/), [Diederik P. Kingma](http://dpkingma.com/), [Abhishek Kumar](http://users.umiacs.umd.edu/~abhishek/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), and [Ben Poole](https://cs.stanford.edu/~poole/)


