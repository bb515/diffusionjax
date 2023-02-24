"""Controllable generation.
"""
from jax import jit, vmap, grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from diffusionjax.plot import (
    plot_samples, plot_score, plot_score_ax, plot_heatmap, plot_animation)
from diffusionjax.losses import get_loss_fn
from diffusionjax.samplers import EulerMaruyama
from diffusionjax.utils import (
    MLP,
    get_score_fn,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.sde import OU


def sample_circle(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples


def main():
    num_epochs = 4000
    rng = random.PRNGKey(2023)
    rng, step_rng = random.split(rng, 2)
    num_samples = 8
    samples = sample_circle(num_samples)
    N = samples.shape[1]
    plot_samples(samples=samples, index=(0, 1), fname="samples", lims=((-3, 3), (-3, 3)))

    # Get sde model
    sde = OU()

    # Neural network training via score matching
    batch_size=16
    score_model = MLP()
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Get loss function
    loss = get_loss_fn(
        sde, score_model, score_scaling=True, likelihood_weighting=False,
        reduce_mean=True, pointwise_t=False)
    # Train with score matching
    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step=update_step,
        num_epochs=num_epochs,
        step_rng=step_rng,
        samples=samples,
        score_model=score_model,
        params=params,
        opt_state=opt_state,
        loss_fn=loss,
        batch_size=batch_size)
    # Get trained score
    trained_score = get_score_fn(sde, score_model, params, score_scaling=True)
    plot_score(score=trained_score, t=0.01, area_min=-3, area_max=3, fname="trained score")
    sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
    q_samples = sampler(rng, n_samples=1000, shape=(N,))
    plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap trained score")

    # Condition on one of the coordinates


if __name__ == "__main__":
    main()