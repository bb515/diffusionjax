"""Utility functions, including all functions related to
loss computation, optimization and sampling.
"""
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap
import jax.random as random
from functools import partial


def batch_linalg_solve_A(A, b):
  return vmap(lambda b: jnp.linalg.solve(A, b))(b)


def batch_linalg_solve(A, b):
  return vmap(jnp.linalg.solve)(A, b)


def batch_mul(a, b):
  return vmap(lambda a, b: a * b)(a, b)


def batch_mul_A(a, b):
  return vmap(lambda b: a * b)(b)


def batch_matmul(A, b):
  return vmap(lambda A, b: A @ b)(A, b)


def errors(t, sde, score, rng, data, likelihood_weighting=True):
  """
  Args:
    ts: JAX array of times.
    sde: Instantiation of a valid SDE class.
    score: A function taking in (x, t) and returning the score.
    rng: Random number generator from JAX.
    data: A batch of samples from the training data, representing samples from the data distribution, shape (J, N).
    likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
  Returns:
    A Monte-Carlo approximation to the (likelihood weighted) score errors.
  """
  m = sde.mean_coeff(t)
  mean = batch_mul(m, data)
  std = jnp.sqrt(sde.variance(t))
  rng, step_rng = random.split(rng)
  noise = random.normal(step_rng, data.shape)
  x = mean + batch_mul(std, noise)
  if not likelihood_weighting:
    return noise + batch_mul(score(x, t), std)
  else:
    return batch_mul(noise, 1. / std) + score(x, t)


def get_loss(sde, solver, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
  """Create a loss function for score matching training.
  Args:
    sde: Instantiation of a valid SDE class.
    solver: Instantiation of a valid Solver class.
    model: A valid flax neural network `:class:flax.linen.Module` class.
    score_scaling: Bool, set to `True` if learning a score scaled by the marginal standard deviation.
    likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    reduce_mean: Bool, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
    pointwise_t: Bool, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

  Returns:
    A loss function that can be used for score matching training.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  if pointwise_t:
    def pointwise_loss(t, params, rng, data):
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
    return pointwise_loss
  else:
    def loss(params, rng, data):
      rng, step_rng = random.split(rng)
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


def get_score(sde, model, params, score_scaling):
  if score_scaling is True:
    return lambda x, t: -batch_mul(model.apply(params, x, t), 1. / jnp.sqrt(sde.variance(t)))
  else:
    return lambda x, t: -model.apply(params, x, t)


def get_epsilon(sde, model, params, score_scaling):
  if score_scaling is True:
    return lambda x, t: model.apply(params, x, t)
  else:
    return lambda x, t: batch_mul(jnp.sqrt(sde.variance(t)), model.apply(params, x, t))


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
    denoise: Bool, that if `True` applies one-step denoising to final samples.
    stack_samples: Bool, that if `True` return all of the sample path or
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
          `x_T sim Normal(O, I)` in reverse-time diffusion.
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
        vec_t = jnp.full((shape[0],), t)
        rng, step_rng = random.split(rng)
        x, x_mean = outer_update(step_rng, x, vec_t)
        if not stack_samples:
          return (rng, x, x_mean), ()
        else:
          return ((rng, x, x_mean), x_mean) if denoise else ((rng, x, x_mean), x)

    rng, step_rng = random.split(rng)
    if x_0 is None:
      x = outer_solver.prior(step_rng, shape)
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
