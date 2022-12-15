"""Score based generative models introduction.

Based off the Jupyter notebook: https://jakiw.com/sgm_intro
A tutorial on the theoretical and implementation aspects of score-based generative models, also called diffusion models.
"""
from jax import jit, vmap, grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from sgm.plot import (
    plot_samples, plot_score, plot_heatmap, plot_animation)
from sgm.losses import get_loss_fn
from sgm.samplers import EulerMaruyama
from sgm.utils import (
    MLP,
    CNN,
    get_score_fn,
    update_step,
    optimizer,
    retrain_nn)
from sgm.sde import get_sde
from mlkernels import EQ, Matern52
from jax.scipy.linalg import solve_triangular
from numpy.linalg import cholesky
import numpy as np
import lab as B
from functools import partial


def batch_mul(a, b):
    return vmap(lambda a, b: a * b)(a, b)


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
    img = image_grid(x, image_size, num_channels)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(fname)
    plt.close()


def sample_image_rgb(rng, num_samples, image_size, kernel, num_channels):
    """Samples from a GMRF
    """
    x = np.linspace(-10e0, 10e0, image_size)
    y = np.linspace(-10e0, 10e0, image_size)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(image_size**2, 1)
    yy = yy.reshape(image_size**2, 1)
    z = np.hstack((xx, yy))
    C_0 = jnp.asarray(B.dense(kernel(z)))
    z = random.normal(rng, (num_samples, xx.shape[0], num_channels))
    x = jnp.einsum('ij, kil -> kjl', cholesky(C_0), z)  # (n_samples, image_size**2, num_channels)
    return x


def plot_score_ax_sample(ax, sample, score, t, area_min=-1, area_max=1, fname="plot_score"):
    @partial(jit, static_argnums=[0,])
    def helper(score, sample, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        sample = jnp.tile(sample, (len(x.flatten()), 1, 1, 1))
        sample.at[:, [0, 1], 0, 0].set(grid)
        t = jnp.ones((grid.shape[0],)) * t
        scores = score(sample, t)
        return grid, scores
    grid, scores = helper(score, sample, t, area_min, area_max)
    ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0, 0, 0], scores[:, 1, 0, 0])


def main():
    num_epochs = 1000
    rng = random.PRNGKey(2023)
    rng, step_rng = random.split(rng, 2)
    num_samples = 64
    num_channels = 1
    image_size = 32  # image size

    samples = sample_image_rgb(rng, num_samples=num_samples, image_size=image_size, kernel=Matern52(), num_channels=num_channels)  # (num_samples, image_size**2, num_channels)
    for i in range(num_samples):
        if i % 2**7 == 0:
            plot_samples(samples[i], image_size=image_size, num_channels=num_channels)
    # Reshape image data
    samples = samples.reshape(-1, image_size, image_size, num_channels)

    # Get sde model
    sde = get_sde("OU")

    def log_hat_pt_tmp(x, t):
        """
        Empirical distribution score for normal distribution on the hyperplane.

        Args:
            x: One location in $\mathbb{R}^2$
            t: time
        Returns:
            The empirical log density, as described in the Jupyter notebook
            .. math::
                \hat{p}_{t}(x)
        """
        mean, std = sde.marginal_prob(samples[:, [0, 1], 0, 0], t)
        potentials = jnp.sum(-(x - mean)**2 / (2 * std**2), axis=1)
        return logsumexp(potentials, axis=0, b=1/num_samples)

    def log_hat_pt(x, t):
        """
        Empirical distribution score for normal distribution on the hyperplane.

        Args:
            x: One location in $\mathbb{R}^2$
            t: time
        Returns:
            The empirical log density, as described in the Jupyter notebook
            .. math::
                \hat{p}_{t}(x)
        """
        mean, std = sde.marginal_prob(samples, t)
        losses = -(x - mean)**2 / (2 * std**2)
        # Needs to be reshaped, since x is an image
        potentials = jnp.sum(losses.reshape((losses.shape[0], -1)), axis=-1)
        return logsumexp(potentials, axis=0, b=1/num_samples)

    # Get a jax grad function, which can be batched with vmap
    nabla_log_hat_pt_tmp = jit(vmap(grad(log_hat_pt_tmp), in_axes=(0, 0), out_axes=(0)))
    nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0), out_axes=(0)))

    # Running the reverse SDE with the empirical drift
    plot_score(score=nabla_log_hat_pt_tmp, t=0.01, area_min=-3, area_max=3, fname="empirical score")
    sampler = EulerMaruyama(sde, nabla_log_hat_pt).get_sampler()
    q_samples = sampler(rng, n_samples=64, shape=(image_size, image_size, num_channels))
    plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples empirical score")
    plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_min=-3, area_max=3, fname="heatmap empirical score")

    # What happens when I perturb the score with a constant?
    perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 100.0 * jnp.ones(jnp.shape(x))
    rng, step_rng = random.split(rng)
    sampler = EulerMaruyama(sde, perturbed_score).get_sampler()
    q_samples = sampler(rng, n_samples=64, shape=(image_size, image_size, num_channels))
    plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples bounded perturbation")
    plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_min=-3, area_max=3, fname="heatmap bounded perturbation")

    # Neural network training via score matching
    batch_size = 32
    score_model = CNN()
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, image_size, image_size, num_channels)), jnp.ones((batch_size,)))
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
    sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
    q_samples = sampler(rng, n_samples=64, shape=(image_size, image_size, num_channels))
    plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples trained score")
    plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_min=-3, area_max=3, fname="heatmap trained score")

    frames = 100
    fig, ax = plt.subplots()
    def animate(i, ax):
        ax.clear()
        plot_score_ax_sample(
            ax, q_samples[0], trained_score, t=1 - (i / frames), area_min=-5, area_max=5, fname="trained score")
    # Plot animation of the trained score over time
    plot_animation(fig, ax, animate, frames, "trained_score")


if __name__ == "__main__":
    main()

