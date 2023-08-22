"""Utility functions, including all functions related to
loss computation, optimization, sampling and inverse problems.
"""
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, jit, value_and_grad
import jax.random as random
from functools import partial
import optax
import numpy as np


def batch_mul(a, b):
    return vmap(lambda a, b: a * b)(a, b)


def errors(ts, sde, score, rng, data, likelihood_weighting=True):
    """
    Args:
        ts: JAX array of times.
        sde: Instantiation of a valid SDE class.
        score: A function taking in (x, t) and returning the score.
        rng: Random number generator from JAX.
        data: A batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N).
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    Returns:
        A random (MC) approximation to the (likelihood weighted) score errors.
    """
    mean, std = sde.marginal_prob(data, ts)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    x_t = mean + batch_mul(std, noise)
    if not likelihood_weighting:
        return noise + batch_mul(score(x_t, ts), std)
    else:
        return batch_mul(noise, 1. / std) + score(x_t, ts)


def get_loss(sde, solver, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for score matching training.
    Args:
        sde: Instantiation of a valid SDE class.
        solver: Instantiation of a valid Solver class.
        model: A valid flax neural network `:class:flax.linen.Module` class.
        score_scaling: Boolean variable, set to `True` if learning a score scaled by the marginal standard deviation.
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
        reduce_mean: Boolean variable, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
        pointwise_t: Boolean variable, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

    Returns:
        A loss function that can be used for score matching training.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss(t, params, rng, data):
            n_batch = data.shape[0]
            ts = jnp.ones((n_batch,)) * t
            score = get_score(sde, model, params, score_scaling)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
                losses = losses * g2
            return jnp.mean(losses)
    else:
        def loss(params, rng, data):
            rng, step_rng = random.split(rng)
            n_batch = data.shape[0]
            ts = random.uniform(step_rng, (data.shape[0],), minval=solver.ts[0], maxval=solver.t1)
            score = get_score(sde, model, params, score_scaling)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
                losses = losses * g2
            return jnp.mean(losses)
    return loss


def get_step_fn(loss, optimizer, train, pmap):
    """Create a one-step training/evaluation function.

    Args:
        loss: A loss function.
        optimizer: An optimization function.
        train: `True` for training and `False` for evaluation.
        pmap: `True` for pmap across jax devices, `False` for single device.

    Returns:
        A one-step function for training or evaluation.
    """
    @jit
    def step_fn(carry, batch):
        (rng, params, opt_state) = carry
        rng, step_rng = random.split(rng)
        grad_fn = value_and_grad(loss)
        if train:
            loss_val, grads = grad_fn(params, step_rng, batch)
            if pmap:
                loss_val = jax.lax.pmean(loss_val, axis_name='batch')
                grads = jax.lax.pmean(grads, axis_name='batch')
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        else:
            loss_val = loss(params, step_rng, batch)
            if pmap: loss_val = jax.lax.pmean(loss_val, axis_name='batch')
        return (rng, params, opt_state), loss_val
    return step_fn


#Initialize the optimizer
optimizer = optax.adam(1e-3)


@partial(jit, static_argnums=[4])
def update_step(params, rng, batch, opt_state, loss):
    """
    Takes the gradient of the loss function and updates the model weights (params) using it.
    Args:
        params: the current weights of the model
        rng: random number generator from jax
        batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
        opt_state: the internal state of the optimizer
        loss: A loss function that can be used for score matching training.
    Returns:
        The value of the loss function (for metrics), the new params and the new optimizer states function (for metrics),
        the new params and the new optimizer state.
    """
    val, grads = value_and_grad(loss)(params, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


def retrain_nn(
        update_step, num_epochs, step_rng, samples, params,
        opt_state, loss, batch_size=5):
    train_size = samples.shape[0]
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    mean_losses = jnp.zeros((num_epochs, 1))
    for i in range(num_epochs):
        rng, step_rng = random.split(step_rng)
        perms = random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = jnp.zeros((jnp.shape(perms)[0], 1))
        for j, perm in enumerate(perms):
            batch = samples[perm, :]
            rng, step_rng = random.split(rng)
            loss_eval, params, opt_state = update_step(params, step_rng, batch, opt_state, loss)
            losses = losses.at[j].set(loss_eval)
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
    return params, opt_state, mean_losses


def get_score(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: -batch_mul(model.apply(params, x, t), 1. / sde.marginal_prob(x, t)[1])
    else:
        return lambda x, t: -model.apply(params, x, t)


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
    if inverse_scaler is None: inverse_scaler = lambda x: x

    def sampler(rng, x_0=None):
        """
        Args:
            rng: A JAX random state.
            x_0: Initial condition. If `None`, then samples an initial condition from the
                sde's initial condition prior. Note that this initial condition represents
                `x_T \sim \text{Normal}(O, I)` in reverse-time diffusion.
        Returns:
            Samples and the number of score function (model) evaluations.
        """
        outer_update = partial(shared_update,
                                         solver=outer_solver)
        outer_ts = outer_solver.ts

        if inner_solver:
            inner_update = partial(shared_update,
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


def merge_data_with_mask(x_space, data, mask, coeff=1.):
    return data * mask * coeff + x_space * (1. - mask * coeff)


def get_projection_sampler(solver, inverse_scaler=None, denoise=True, stack_samples=False):
    """Create an image inpainting function that uses sampler, that returns only the final sample.

    Args:
        inverse_scaler: The inverse data normalizer.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
        stack_samples: Boolean variable that if `True` returns all samples on path(s).
    Returns:
        A projection sampler function.
    """
    if inverse_scaler is None:
        def inverse_scaler(x):
            return x
    vmap_inverse_scaler = vmap(inverse_scaler)

    def update(rng, data, mask, x, vec_t, coeff):
        data_mean, std = solver.sde.marginal_prob(data, vec_t)
        z = random.normal(rng, x.shape)
        z_data = data_mean + batch_mul(std, z)
        x = merge_data_with_mask(x, z_data, mask, coeff)

        rng, step_rng = random.split(rng)
        x, x_mean = solver.update(step_rng, x, vec_t)
        return x, x_mean

    ts = solver.ts
    num_function_evaluations = jnp.size(ts)

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
            vec_t = jnp.full(shape[0], t)
            rng, step_rng = random.split(rng)
            x, x_mean = update(step_rng, data, mask, x, vec_t, coeff)
            return ((rng, x, x_mean), x) if stack_samples else ((rng, x, x_mean), ())

        if not stack_samples:
            (_, x, _), _ = scan(f, (rng, x, x), ts, reverse=True)
            return inverse_scaler(x), num_function_evaluations
        else:
            (_, _, _), xs = scan(f, (rng, x, x), ts, reverse=True)
            return vmap_inverse_scaler(xs), num_function_evaluations

    # return jax.pmap(projection_sampler, in_axes=(0, None, None, None), axis_name='batch')
    return projection_sampler


def get_inpainter(solver, inverse_scaler=None,
                denoise=True, stack_samples=False):
    """Create an image inpainting function that uses sampler, that returns only the final sample.

    Args:
        inverse_scaler: The inverse data normalizer.
        denoise: Boolean variable that if `True` applies one-step denoising to final samples.
    Returns:
        An inpainting function.
    """
    if inverse_scaler is None:
        def inverse_scaler(x):
            return x
    vmap_inverse_scaler = vmap(inverse_scaler)

    def update(rng, data, mask, x, vec_t):
        x, x_mean = solver.update(rng, x, vec_t)
        masked_data_mean, std = solver.sde.marginal_prob(data, vec_t)
        rng, step_rng = random.split(rng)
        masked_data = masked_data_mean + batch_mul(random.normal(rng, x.shape), std)
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    ts = solver.ts
    num_function_evaluations = jnp.size(ts)

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
            vec_t = jnp.full(shape[0], t)
            rng, step_rng = random.split(rng)
            x, x_mean = update(rng, data, mask, x, vec_t)
            return ((rng, x, x_mean), x) if stack_samples else ((rng, x, x_mean), ())

        if not stack_samples:
            (_, x, _), _ = scan(f, (rng, x, x), ts, reverse=True)
            return inverse_scaler(x), num_function_evaluations
        else:
            (_, _, _), xs = scan(f, (rng, x, x), ts, reverse=True)
            return vmap_inverse_scaler(xs), num_function_evaluations
    # return jax.pmap(inpainter, in_axes=(0, None, None), axis_name='batch')
    return inpainter
