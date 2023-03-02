"""Samplers."""
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random


def get_sampler(outer_solver, inner_solver=None, denoise=True, stack_samples=False):
    """Get a sampler from (possibly interleaved) numerical solver(s).

    Args:
        outer_solver: A valid numerical solver class that will act on an outer loop.
        inner_solver: "" "" that will act on an inner loop.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
        stack_samples: Boolean variable that if `True` return all of the sample path or
            just returns the last sample.
    Returns:
        A sampler.
    """

    def sampler(rng, n_samples, shape, x_0=None):
        """

        Args:
            rng:
            n_samples:
            shape:
            x_0: Initial condition. If `None`, then samples an initial condition from the
                sde's initial condition prior.
        Returns:
            Samples.
        """
        outer_update = outer_solver.get_update()
        outer_ts = outer_solver.ts

        if inner_solver:
            inner_update = inner_solver.get_update()
            inner_ts = inner_solver.ts
            def inner_step(carry, t):
                rng, x, x_mean, vec_t = carry
                rng, step_rng = random.split(rng)
                x, x_mean = inner_update(step_rng, x, vec_t)
                return (rng, x, x_mean, vec_t), ()

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, x_mean = outer_update(step_rng, x, vec_t)
                (rng, x, x_mean, vec_t), _ = scan(inner_step, (step_rng, x, x_mean, vec_t), inner_ts)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    if denoise:
                        return (rng, x, x_mean), x_mean
                    else:
                        return (rng, x, x_mean), x
        else:
            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, x_mean = outer_update(step_rng, x, vec_t)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    if denoise:
                        return (rng, x, x_mean), x_mean
                    else:
                        return (rng, x, x_mean), x

        rng, step_rng = random.split(rng)
        n_samples_shape = (n_samples,) + shape
        if x_0 is None:
            x = random.normal(step_rng, n_samples_shape)
        else:
            assert(x_0.shape==n_samples_shape)
            x = x_0
        if not stack_samples:
            (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts)
            if denoise:
                return x_mean
            else:
                return x
        else:
            (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts)
            return xs
    return sampler


def get_augmented_sampler(outer_solver, inner_solver=None, stack_samples=False):
    """Get a sampler, with augmented state-space (position and velocity), from (possibly interleaved) numerical solver(s).

    Args:
        outer_solver: A valid numerical solver class that will act on an outer loop.
        inner_solver: "" "" that will act on an inner loop.
        stack_samples:
    Returns:
        A sampler.
    """

    outer_ts = outer_solver.ts
    outer_update = outer.get_update()

    def sampler(rng, n_samples, shape, x_0=None, xd_0=None):
        """

        Args:
            rng:
            n_samples:
            shape:
            x_0:
            xd_0:
        Returns:
            Samples.
        """
        if inner_solver:
            inner_update = inner_solver.get_update()
            inner_ts = inner_solver.ts
            def inner_step(carry, t):
                rng, x, xd, vec_t = carry
                rng, step_rng = random.split(rng)
                x, xd, xd_mean = inner_update(step_rng, x, xd, vec_t)
                return (rng, x, xd, vec_t), ()

            def outer_step(carry, t):
                rng, x, xd, xd_mean = carry
                vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, xd = outer_update(step_rng, x, xd, vec_t)
                (rng, x, xd, xd_mean), _ = scan(
                    inner_step, (step_rng, x, xd, vec_t), inner_ts)
                if not stack_samples:
                    return (rng, x, xd), ()
                else:
                    return (rng, x, xd), x
        else:
            def outer_step(carry, t):
                rng, x, xd, xd_mean = carry
                vec_t = jnp.ones((n_samples, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, xd = outer_update(step_rng, x, xd, vec_t)
                if not stack_samples:
                    return (rng, x, xd), ()
                else:
                    return (rng, x, xd), x

        rng, step_rng = random.split(rng)
        n_samples_shape = (n_samples,) + shape
        if x_0 is None:
            x = random.normal(step_rng, n_samples_shape)
            xd = random.normal(step_rng, n_samples_shape)
        else:
            assert(x_0.shape==n_samples_shape)
            assert(xd_0.shape==n_samples_shape)
            x = x_0
            xd = xd_0
        if not stack_samples:
            (_, x, xd, _), _ = scan(outer_step, (rng, x, xd, xd), outer_ts)
            return x, xd
        else:
            (_, _, _, _), xs = scan(outer_step, (rng, x, xd, xd), outer_ts)
            return xs
    return sampler
