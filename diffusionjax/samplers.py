"""Sampler classes."""
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from jax import vmap
from sgm.utils import batch_mul


class EulerMaruyama():
    """Euler Maruyama numerical solver of an SDE.Functions are designed for a mini-batch of inputs."""

    def __init__(self, sde, score_fn):
        """Constructs an Euler Maruyama sampler.
        Args:
            sde: A valid SDE class.
            score_fn: A valid score function.
        """
        self.sde = sde
        # Compute the reverse sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn

    def get_update(self):
        discretize = self.rsde.discretize

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

    def get_sampler_enumerate(self):
        ts = self.sde.train_ts
        update = self.get_update()
        def sampler(rng, n_samples, shape):
            rng, step_rng = random.split(rng)
            n_samples_shape = (n_samples,) + shape
            print(n_samples_shape)
            x = random.normal(step_rng, n_samples_shape)
            def f(carry, t):
                rng, x, x_mean, i = carry
                vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, x_mean = update(rng, x, vec_t)
                i += 1
                return (rng, x, x_mean, i), ()
            (_, x, _), _ = scan(f, (rng, x, x, 0), ts)
            return x

    def get_sampler(self, stack_samples=False):
        ts = self.sde.train_ts
        update = self.get_update()
        if not stack_samples:

            def sampler(rng, n_samples, shape):
                rng, step_rng = random.split(rng)
                n_samples_shape = (n_samples,) + shape
                x = random.normal(step_rng, n_samples_shape)
                def f(carry, t):
                    rng, x, x_mean = carry
                    vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                    rng, step_rng = random.split(rng)
                    x, x_mean = update(rng, x, vec_t)
                    return (rng, x, x_mean), ()
                (_, x, _), _ = scan(f, (rng, x, x), ts)
                return x
        else:

            def sampler(rng, n_samples, shape, stack_samples=False):
                rng, step_rng = random.split(rng)
                n_samples_shape = (n_samples) + shape
                x = random.normal(step_rng, n_samples_shape)
                def f(carry, t):
                    rng, x, x_mean = carry
                    vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                    rng, step_rng = random.split(rng)
                    x, x_mean = update(rng, x, t)
                    return (rng, x, x_mean), x
                (_, _, _), xs = scan(f, (rng, x, x), ts)
                return xs
        return sampler

