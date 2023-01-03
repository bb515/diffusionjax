"""Score based generative models introduction.

An example using 2 dimensional image data.
"""
from jax import jit, vmap, grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from sgm.plot import plot_samples, plot_heatmap
from sgm.losses import get_loss_fn
from sgm.samplers import EulerMaruyama
from sgm.utils import (
    CNN,
    get_score_fn,
    update_step,
    optimizer,
    retrain_nn)
from sgm.sde import OU
from mlkernels import Matern52
import numpy as np
import lab as B


x_max = 5.0
epsilon = 1e-4


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
    """Samples from a GMRF."""
    x = np.linspace(-x_max, x_max, image_size)
    y = np.linspace(-x_max, x_max, image_size)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(image_size**2, 1)
    yy = yy.reshape(image_size**2, 1)
    z = np.hstack((xx, yy))
    C = B.dense(kernel(z))  + epsilon * B.eye(image_size**2)
    x = random.multivariate_normal(rng, mean=jnp.zeros(xx.shape[0]), cov=C, shape=(num_samples, num_channels))
    x = x.transpose((0, 2, 1))
    return x, C


def plot_samples_1D(samples, image_size, fname="samples 1D.png"):
    x = np.linspace(-x_max, x_max, image_size)
    plt.plot(x, samples[:, :, 0, 0].T)
    plt.savefig(fname)
    plt.close()


def main():
    num_epochs = 200
    rng = random.PRNGKey(2023)
    rng, step_rng = random.split(rng, 2)
    num_samples = 144
    num_channels = 1
    image_size = 32  # image size

    samples, C = sample_image_rgb(rng, num_samples=num_samples, image_size=image_size, kernel=Matern52(), num_channels=num_channels)  # (num_samples, image_size**2, num_channels)
    plot_samples(samples, image_size=image_size, num_channels=num_channels)
    # Reshape image data
    samples = samples.reshape(-1, image_size, image_size, num_channels)
    plot_samples_1D(samples, image_size, "samples 1D")

    # Get sde model
    sde = OU()

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
        losses = -(x - mean)**2 / (2 * std**2)
        # Needs to be reshaped, since x is an image
        potentials = jnp.sum(losses.reshape((losses.shape[0], -1)), axis=-1)
        return logsumexp(potentials, axis=0, b=1/num_samples)

    def nabla_log_pt(x, t):
        """
        Args:
            x: One location in $\mathbb{R}^2$
            t: time
        Returns:
            The true log density.
            .. math::
                p_{t}(x)
        """
        x_shape = x.shape
        v_t = sde.variance(t)
        m_t = sde.mean_coeff(t)
        x = x.flatten()
        score = - jnp.linalg.solve(m_t**2 * C + v_t * jnp.eye(x_shape[0] * x_shape[1]), x)
        return score.reshape(x_shape)

    # Get a jax grad function, which can be batched with vmap
    nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0), out_axes=(0)))
    nabla_log_pt = jit(vmap(nabla_log_pt, in_axes=(0, 0), out_axes=(0)))
    # Running the reverse SDE with the empirical score
    sampler = EulerMaruyama(sde, nabla_log_hat_pt).get_sampler()
    q_samples = sampler(rng, n_samples=64, shape=(image_size, image_size, num_channels))
    plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples empirical score")
    plot_samples_1D(q_samples, image_size, "samples 1D empirical score")
    plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_min=-3, area_max=3, fname="heatmap empirical score")
    # What happens when I perturb the score with a constant?
    perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 10.0 * jnp.ones(jnp.shape(x))
    rng, step_rng = random.split(rng)
    sampler = EulerMaruyama(sde, perturbed_score).get_sampler()
    q_samples = sampler(rng, n_samples=64, shape=(image_size, image_size, num_channels))
    plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples bounded perturbation")
    plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_min=-3, area_max=3, fname="heatmap bounded perturbation")

    # Neural network training via score matching
    batch_size = 16
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
    rng, step_rng = random.split(rng, 2)
    q_samples = sampler(rng, n_samples=64, shape=(image_size, image_size, num_channels))
    plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples trained score")
    plot_samples_1D(q_samples, image_size, fname="samples 1D trained score")
    plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_min=-3, area_max=3, fname="heatmap trained score")


if __name__ == "__main__":
    main()

