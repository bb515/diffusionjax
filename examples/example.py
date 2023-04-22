"""Diffusion models introduction.

Based off the Jupyter notebook: https://jakiw.com/sgm_intro
A tutorial on the theoretical and implementation aspects of score-based generative models, also called diffusion models.
"""
from jax import jit, vmap, grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import serialization
import matplotlib.pyplot as plt
from diffusionjax.plot import (
    plot_samples, plot_score, plot_score_ax, plot_heatmap, plot_heatmap_ax,
    plot_animation)
from diffusionjax.losses import get_loss
from diffusionjax.solvers import EulerMaruyama, Annealed
from diffusionjax.samplers import get_sampler
from diffusionjax.inverse_problems import get_inpainter
from diffusionjax.models import MLP
from diffusionjax.utils import (
    get_score,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.sde import VP, VE, UDLangevin


def sample_circle(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.
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

    # Get sde model, variance preserving (VP) a.k.a. time-changed Ohrnstein Uhlenbeck (OU)
    sde = VP(beta_min=0.01, beta_max=3.0)
    # Alternative sde model, variance exploding (VE) a.k.a. time-dependent diffusion process
    # sde = VE(sigma_min=0.01, sigma_max=3.0)

    def log_hat_pt(x, t):
        """
        Empirical distribution score.

        Args:
            x: One location in $\mathbb{R}^2$
            t: time
        Returns:
            The empirical log density, as described in the Jupyter notebook
            .. math::
                \hat{p}_{t}(x)
        """
        mean, std = sde.marginal_prob(samples, t)
        potentials = jnp.sum(-(x - mean)**2 / (2 * std**2), axis=1)
        return logsumexp(potentials, axis=0, b=1/num_samples)

    # Get a jax grad function, which can be batched with vmap
    nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0), out_axes=(0)))

    # Running the reverse SDE with the empirical drift
    plot_score(score=nabla_log_hat_pt, t=0.01, area_min=-3, area_max=3, fname="empirical score")
    reverse_sde = sde.reverse(nabla_log_hat_pt)
    solver = EulerMaruyama(reverse_sde)
    sampler = get_sampler(solver, stack_samples=False)
    q_samples = sampler(rng, num_samples=5000, shape=(N,))
    plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap empirical score")

    # What happens when I perturb the score with a constant?
    perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 1
    rng, step_rng = random.split(rng)
    reverse_sde = sde.reverse(perturbed_score)
    solver = EulerMaruyama(reverse_sde)
    sampler = get_sampler(solver)
    q_samples = sampler(rng, num_samples=5000, shape=(N,))
    plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap bounded perturbation")

    # Neural network training via score matching
    batch_size=16
    score_model = MLP()
    score_scaling = True
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    if 0:  # Load pre-trained model parameters
        f = open('/tmp/output0', 'rb')
        output = f.read()
        params = serialization.from_bytes(params, output)
    else:
        # Get loss function
        solver = EulerMaruyama(sde)
        loss = get_loss(
            sde, solver, score_model, score_scaling=score_scaling, likelihood_weighting=False,
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
            loss=loss,
            batch_size=batch_size)

        # Save params
        output = serialization.to_bytes(params)
        f = open('/tmp/output0', 'wb')
        f.write(output)

    # Get trained score
    trained_score = get_score(sde, score_model, params, score_scaling=score_scaling)
    plot_score(score=trained_score, t=0.01, area_min=-3, area_max=3, fname="trained score")
    reverse_sde = sde.reverse(trained_score)
    solver = EulerMaruyama(reverse_sde)
    sampler = get_sampler(solver, stack_samples=False)
    q_samples = sampler(rng, num_samples=1000, shape=(N,))
    plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap trained score")

    # Condition on one of the coordinates
    data = jnp.array([-0.5, 0.0])
    mask = jnp.array([1, 0])
    data = jnp.tile(data, (64, 1))
    mask = jnp.tile(mask, (64, 1))
    inpainter = get_inpainter(solver, stack_samples=False)
    q_samples = inpainter(rng, data, mask)
    plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname="heatmap inpainted")

    frames = 100
    fig, ax = plt.subplots(1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    def animate(i, ax):
        ax.clear()
        plot_score_ax(
            ax, trained_score, t=1 - (i / frames), area_min=-1, area_max=1)

    # Plot animation of the trained score over time
    plot_animation(fig, ax, animate, frames, "trained_score")
    plt.close()

if __name__ == "__main__":
    main()
