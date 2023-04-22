"""Inverse problems."""
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
from diffusionjax.utils import batch_mul
from jax.experimental.host_callback import id_print
import numpy as np


def merge_data_with_mask(x_space, data, mask, coeff=1.):
    return data * mask * coeff + x_space * (1. - mask * coeff)


def get_projection_sampler(solver, inverse_scaler=None, denoise=True, stack_samples=False):
    """Create an image inpainting function that uses sampler, that returns only the final sample.

    Args:
        inverse_scaler: The inverse data normalizer.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
        stack_samples: Boolean variable that if `True` returns all samples on path(s).
    Returns:
        A pmapped inpainting function.
    """
    def update(rng, data, mask, x, vec_t, coeff):
        data_mean, std = solver.sde.marginal_prob(data, vec_t)
        z = random.normal(rng, x.shape)
        z_data = data_mean + batch_mul(std, z)
        x = merge_data_with_mask(x, z_data, mask, coeff)

        rng, step_rng = random.split(rng)
        x, x_mean = solver.update(step_rng, x, vec_t)
        return x, x_mean

    ts = solver.ts

    def projection_sampler(rng, data, mask, coeff):
        """Sampler for image inpainting.

        Args:
            rng: A JAX random state.
            data: A JAX array that represents a mini-batch of images to inpaint.
            mask: A {0, 1} array with the same shape of `data`. Value `1` marks known pixels,
                and value `0` marks pixels that require inpainting.

        Returns:
            Inpainted (complete) images.
        """
        # Initial sample
        rng, step_rng = random.split(rng)
        shape = data.shape
        x = random.normal(step_rng, shape)

        def f(carry, t):
            rng, x, x_mean = carry
            vec_t = jnp.ones((shape[0], 1)) * (1. - t)
            rng, step_rng = random.split(rng)
            x, x_mean = update(step_rng, data, mask, x, vec_t, coeff)
            if not stack_samples:
                return (rng, x, x_mean), ()
            else:
                return (rng, x, x_mean), x
        if not stack_samples:
            (_, x, _), _ = scan(f, (rng, x, x), ts)
            return x
        else:
            (_, _, _), xs = scan(f, (rng, x, x), ts)
            return xs

    return projection_sampler  # TODO: pmap axis_name="batch"


def get_inpainter(solver, inverse_scaler=None,
                denoise=True, stack_samples=False):
    """Create an image inpainting function that uses sampler, that returns only the final sample.

    Args:
        inverse_scaler: The inverse data normalizer.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
    Returns:
        A pmapped inpainting function.
    """
    def update(rng, data, mask, x, vec_t):
        x, x_mean = solver.update(rng, x, vec_t)
        masked_data_mean, std = solver.sde.marginal_prob(data, vec_t)
        rng, step_rng = random.split(rng)
        masked_data = masked_data_mean + batch_mul(random.normal(rng, x.shape), std)
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    ts = solver.ts

    def inpainter(rng, data, mask):
        """Sampler for image inpainting.

        Args:
            rng: A JAX random state.
            data: A JAX array that represents a mini-batch of images to inpaint.
            mask: A {0, 1} array with the same shape of `data`. Value `1` marks known pixels,
                and value `0` marks pixels that require inpainting.

        Returns:
            Inpainted (complete) images.
        """
        # Initial sample
        rng, step_rng = random.split(rng)
        shape = data.shape
        x = data * mask + random.normal(step_rng, shape) * (1. - mask)

        def f(carry, t):
            rng, x, x_mean = carry
            vec_t = jnp.ones((shape[0], 1)) * (1. - t)
            rng, step_rng = random.split(rng)
            x, x_mean = update(rng, data, mask, x, vec_t)
            if not stack_samples:
                return (rng, x, x_mean), ()
            else:
                return (rng, x, x_mean), x
        if not stack_samples:
            (_, x, _), _ = scan(f, (rng, x, x), ts)
            return x
        else:
            (_, _, _), xs = scan(f, (rng, x, x), ts)
            return xs
    return inpainter  # TODO: pmap axis_name="batch"
