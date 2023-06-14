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


def get_sampler(shape, outer_solver, inner_solver=None, denoise=True, stack_samples=False, inverse_scaler=None):
    """Get a sampler from (possibly interleaved) numerical solver(s).

    Args:
        shape: Shape of array, x. (num_samples,) + obj_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            obj_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
        outer_solver: A valid numerical solver class that will act on an outer loop.
        inner_solver: '' that will act on an inner loop.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
        stack_samples: Boolean variable that if `True` return all of the sample path or
            just returns the last sample.
        inverse_scaler: The inverse data normalizer function.
    Returns:
        A sampler.
    """
    if inverse_scaler is None:
        def inverse_scaler(x):
            return x

    def sampler(rng, x_0=None):
        """

        Args:
            rng: A JAX random state.
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
            num_function_evaluations = jnp.size(outer_ts) * (jnp.size(inner_ts) + 1)

            def inner_step(carry, t):
                rng, x, x_mean, vec_t = carry
                rng, step_rng = random.split(rng)
                x, x_mean = inner_update(step_rng, x, vec_t)
                return (rng, x, x_mean, vec_t), ()

            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.full(shape[0], t)
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
            num_function_evaluations = jnp.size(outer_ts)
            def outer_step(carry, t):
                rng, x, x_mean = carry
                vec_t = jnp.full(shape[0], t)
                rng, step_rng = random.split(rng)
                x, x_mean = outer_update(step_rng, x, vec_t)
                if not stack_samples:
                    return (rng, x, x_mean), ()
                else:
                    return ((rng, x, x_mean), x_mean) if denoise else ((rng, x, x_mean), x)

        rng, step_rng = random.split(rng)
        if x_0 is None:
            x = outer_solver.sde.prior(step_rng, shape)
        else:
            assert(x_0.shape==shape)
            x = x_0
        if not stack_samples:
            (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts, reverse=True)

            return inverse_scaler(x_mean if denoise else x), num_function_evaluations
        else:
            (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
            return inverse_scaler(xs), num_function_evaluations
    # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
    return sampler
