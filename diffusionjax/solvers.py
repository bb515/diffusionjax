"""Solver classes."""
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from diffusionjax.utils import batch_mul
import abc


class Solver(abc.ABC):
    """Solver abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, num_steps=1000, dt=None):
        """Construct an SDE.
        Args:
            num_steps: number of discretization time steps.
            dt: time step duration, float or `None`.
                Optional, if provided then final time, t1 = dt * num_steps.
        """
        self.num_steps = num_steps
        if dt is None:
            self.dt = 1. / self.num_steps
            self.ts = jnp.linspace(0, 1, num_steps + 1)[:-1].reshape(-1, 1)
        else:
            self.dt = dt
            t1 = dt * num_steps
            self.ts = jnp.linspace(0, t1, num_steps + 1)[:-1].reshape(-1, 1)

    @abc.abstractmethod
    def update(self, rng, x, t):
        r"""Return the drift and diffusion coefficients of the SDE.

        Args:
            rng: A JAX random state.
            x: A JAX array of the state.
            t: JAX array of the time.

        Returns:
            x: A JAX array of the next state.
            x_mean: A JAX array of the next state without noise, for denoising.
        """


class EulerMaruyama(Solver):
    """Euler Maruyama numerical solver of an SDE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, num_steps=1000):
        """Constructs an Euler-Maruyama Solver.
        Args:
            sde: A valid SDE class.
            num_steps: number of discretization time steps.
        """
        super().__init__(num_steps)
        self.sde = sde

    def update(self, rng, x, t):
        """
        Args:
            rng: A JAX random state.
            x: A JAX array representing the current state.
            t: A JAX array representing the current step.

        Returns:
            x: A JAX array of the next state:
            x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        drift, diffusion = self.sde.sde(x, t)
        f = drift * self.dt
        G = diffusion * jnp.sqrt(self.dt)
        noise = random.normal(rng, x.shape)
        x_mean = x + f
        x = x_mean + batch_mul(G, noise)
        return x, x_mean


class Annealed(Solver):
    """Annealed numerical solver of an SDE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, num_steps=2, snr=1e-2):
        """Constructs an Euler Maruyama sampler.
        Args:
            sde: A valid SDE class.
        """
        super().__init__(num_steps)
        self.sde = sde
        self.snr = snr

    def update(self, rng, x, t):
        """
        Args:
            rng: A JAX random state.
            x: A JAX array representing the current state.
            t: A JAX array representing the current step.

        Returns:
            x: A JAX array of the next state:
            x_mean: A JAX array. The next state without random noise. Useful for denoising.
        """
        grad, diffusion = self.sde.sde(x, t)
        grad_norm = jnp.linalg.norm(
            grad.reshape((grad.shape[0], -1)), axis=-1).mean()
        # TODO: implement parallel mean across batches
        # grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
        noise = random.normal(rng, x.shape)
        noise_norm = jnp.linalg.norm(
            noise.reshape((noise.shape[0], -1)), axis=-1).mean()
        # noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
        # TODO: alpha need not be a mini-batch
        alpha = jnp.exp(2 * self.sde.log_mean_coeff(t))
        dt = (self.snr * noise_norm / grad_norm)**2 * 2 * alpha
        f = batch_mul(grad, dt)
        G = batch_mul(diffusion, jnp.sqrt(dt))
        x_mean = x + f
        x = x_mean + batch_mul(G, noise)
        return x, x_mean
