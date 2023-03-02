"""Solver classes."""
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from diffusionjax.utils import batch_mul


class EulerMaruyama():
    """Euler Maruyama numerical solver of an SDE. Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde):
        """Constructs an Euler Maruyama sampler.
        Args:
            sde: A valid SDE class.
        """
        self.sde = sde
        self.ts = sde.ts

    def get_update(self):
        discretize = self.sde.discretize

        def update(rng, x, t):
            """
            Args:
                rng: A JAX random state.
                x: A JAX array representing the current state.
                t: A JAX array representing the current step.
            
            Returns:
                x: A JAX array of the next state:
                x_mean: A JAX array. The next state without random noise. Useful for denoising.
            """
            f, G = discretize(x, t)
            z = random.normal(rng, x.shape)
            x_mean = x + f
            x = x_mean + batch_mul(G, z)
            return x, x_mean
        return update

