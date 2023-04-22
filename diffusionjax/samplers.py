"""Samplers."""
import jax.numpy as jnp
from jax.lax import scan
import jax.random as random
import functools


def shared_update(rng, x, t, solver, probability_flow=None):
    """A wrapper that configures and returns the update function of the solvers.

    :probablity_flow: Placeholder for probability flow ODE (TODO).
    """
    return solver.update(rng, x, t)


def get_sampler(outer_solver, inner_solver=None, denoise=True, stack_samples=False):
    """Get a sampler from (possibly interleaved) numerical solver(s).

    Args:
        outer_solver: A valid numerical solver class that will act on an outer loop.
        inner_solver: '' that will act on an inner loop.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
        stack_samples: Boolean variable that if `True` return all of the sample path or
            just returns the last sample.
    Returns:
        A sampler.
    """

    def sampler(rng, num_samples, shape, x_0=None):
        """

        Args:
            rng: A JAX random state.
            num_samples: Number of samples to return.
            shape: Shape of array, x.
            x_0: Initial condition. If `None`, then samples an initial condition from the
                sde's initial condition prior. Note that this initial condition represents
                `x_T \sim \text{Normal}(O, I)` in reverse-time diffusion.
        Returns:
            Samples.
        """
        outer_update = functools.partial(shared_update,
                                         solver=outer_solver)
        outer_ts = outer_solver.ts

        if inner_solver:
            inner_update = functools.partial(shared_update,
                                            solver=inner_solver)
            inner_ts = inner_solver.ts

            def inner_step(carry, t):
                rng, x, x_mean, vec_t = carry
                rng, step_rng = random.split(rng)
                x, x_mean = inner_update(step_rng, x, vec_t)
                return (rng, x, x_mean, vec_t), ()

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.ones((num_samples, 1)) * (1 - t)
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
                vec_t = jnp.ones((num_samples, 1)) * (1 - t)
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
        num_samples_shape = (num_samples,) + shape
        if x_0 is None:
            x = outer_solver.sde.prior(step_rng, num_samples_shape)
        else:
            assert(x_0.shape==num_samples_shape)
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
        stack_samples: Boolean variable that if `True` return all of the sample path or
            just returns the last sample.
    Returns:
        A sampler.
    """
    outer_update = functools.partial(shared_update,
                                    solver=outer_solver)
    outer_ts = outer_solver.ts

    def sampler(rng, num_samples, shape, x_0=None, xd_0=None):
        """

        Args:
            rng: A JAX random state.
            num_samples: Number of samples to return.
            shape: Shape of array, x.
            x_0: Initial condition position. If `None`, then samples an initial condition from the
                sde's initial condition prior. Note that this initial condition represents
                `x_T \sim \text{Normal}(O, I)` in reverse-time diffusion.
            x_d0: Initial condition velocity. If `None`, then samples an initial condition from the
                sde's initial condition prior. Note that this initial condition represents
                `x_T \sim \text{Normal}(O, I)` in reverse-time diffusion.
        Returns:
            Samples.
        """
        if inner_solver:
            inner_update = functools.partial(shared_update,
                                            solver=inner_solver)
            inner_ts = inner_solver.ts

            def inner_step(carry, t):
                rng, x, xd, vec_t = carry
                rng, step_rng = random.split(rng)
                x, xd, xd_mean = inner_update(step_rng, x, xd, vec_t)
                return (rng, x, xd, vec_t), ()

            def outer_step(carry, t):
                rng, x, xd, xd_mean = carry
                vec_t = jnp.ones((num_samples, 1)) * (1 - t)
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
                vec_t = jnp.ones((num_samples, 1)) * (1 - t)
                rng, step_rng = random.split(rng)
                x, xd = outer_update(step_rng, x, xd, vec_t)
                if not stack_samples:
                    return (rng, x, xd), ()
                else:
                    return (rng, x, xd), x

        rng, step_rng = random.split(rng)
        num_samples_shape = (num_samples,) + shape
        if x_0 is None:
            x, xd = outer_solver.sde.prior(step_rng, num_samples_shape)
        else:
            assert(x_0.shape==num_samples_shape)
            assert(xd_0.shape==num_samples_shape)
            x = x_0
            xd = xd_0
        if not stack_samples:
            (_, x, xd, _), _ = scan(outer_step, (rng, x, xd, xd), outer_ts)
            return x, xd
        else:
            (_, _, _, _), xs = scan(outer_step, (rng, x, xd, xd), outer_ts)
            return xs
    return sampler
