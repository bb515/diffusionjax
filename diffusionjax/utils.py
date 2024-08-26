"""Utility functions, including all functions related to
loss computation, optimization and sampling.
"""

import jax.numpy as jnp
from jax.lax import scan
from jax import vmap
import jax.random as random
from functools import partial
from collections.abc import MutableMapping


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
  items = []
  for name, cfg in nested_dict.items():
    new_key = parent_key + sep + name if parent_key else name
    if isinstance(cfg, MutableMapping):
      items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
    else:
      items.append((new_key, cfg))

  return dict(items)


def get_timestep(t, t0, t1, num_steps):
  return (jnp.rint((t - t0) * (num_steps - 1) / (t1 - t0))).astype(jnp.int32)


def continuous_to_discrete(betas, dt):
  discrete_betas = betas * dt
  return discrete_betas


def get_exponential_sigma_function(sigma_min, sigma_max):
  log_sigma_min = jnp.log(sigma_min)
  log_sigma_max = jnp.log(sigma_max)

  def sigma(t):
    # return sigma_min * (sigma_max / sigma_min)**t  # Has large relative error close to zero compared to alternative, below
    return jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))

  return sigma


def get_linear_beta_function(beta_min, beta_max):
  """Returns:
  Linear beta (cooling rate parameter) as a function of time,
  It's integral multiplied by -0.5, which is the log mean coefficient of the VP SDE.
  """

  def beta(t):
    return beta_min + t * (beta_max - beta_min)

  def mean_coeff(t):
    """..math: exp(-0.5 * \int_{0}^{t} \beta(s) ds)"""
    return jnp.exp(-0.5 * t * beta_min - 0.25 * t**2 * (beta_max - beta_min))

  return beta, mean_coeff


def get_cosine_beta_function(beta_max, offset=0.08):
  """Returns:
  Squared cosine beta (cooling rate parameter) as a function of time,
  It's integral multiplied by -0.5, which is the log mean coefficient of the VP SDE.
  Note: this implementation cannot perfectly replicate https://arxiv.org/abs/2102.09672
    since it deals with a continuous time formulation of beta(t).
  Args:
    offset: https://arxiv.org/abs/2102.09672 "Use a small offset to prevent
    $\beta(t)$ from being too small near
    $t = 0$, since we found that having tiny amounts of noise at the beginning
    of the process made it hard for the network to predict $\epsilon$
    accurately enough"
  """

  def beta(t):
    # clip to max_beta
    return jnp.clip(jnp.sin((t + offset) / (1.0 + offset) * 0.5 * jnp.pi) / (jnp.cos((t + offset) / (1.0 + offset) * 0.5 * jnp.pi) + 1e-5) * jnp.pi * (1.0 / (1.0 + offset)), a_max=beta_max)

  def mean_coeff(t):
    """..math: -0.5 * \int_{0}^{t} \beta(s) ds"""
    return jnp.cos((t + offset) / (1.0 + offset) * 0.5 * jnp.pi)
    # return jnp.cos((t + offset) / (1.0 + offset) * 0.5 * jnp.pi) / jnp.cos(offset / (1.0 + offset) * 0.5 * jnp.pi)

  return beta, mean_coeff


def get_karras_sigma_function(sigma_min, sigma_max, rho=7):
  """
  A sigma function from Algorithm 2 from Karras et al. (2022) arxiv.org/abs/2206.00364

  Returns:
    A function that can be used like `sigmas = vmap(sigma)(ts)` where `ts.shape = (num_steps,)`, see `test_utils.py` for usage.

  Args:
    sigma_min: Minimum standard deviation of forawrd transition kernel.
    sigma_max: Maximum standard deviation of forward transition kernel.
    rho: Order of the polynomial in t (determines both smoothness and growth
      rate).
  """
  min_inv_rho = sigma_min ** (1 / rho)
  max_inv_rho = sigma_max ** (1 / rho)

  def sigma(t):
    # NOTE: is defined in reverse time of the definition in arxiv.org/abs/2206.00364
    return (min_inv_rho + t * (max_inv_rho - min_inv_rho)) ** rho

  return sigma


def get_karras_gamma_function(num_steps, s_churn, s_min, s_max):
  """
  A gamma function from Algorithm 2 from Karras et al. (2022) arxiv.org/abs/2206.00364
  Returns:
    A function that can be used like `gammas = gamma(sigmas)` where `sigmas.shape = (num_steps,)`, see `test_utils.py` for usage.
  Args:
    num_steps:
    s_churn: "controls the overall amount of stochasticity" in Algorithm 2 from Karras et al. (2022)
    [s_min, s_max] : Range of noise levels that "stochasticity" is enabled.
  """

  def gamma(sigmas):
    gammas = jnp.where(sigmas <= s_max, min(s_churn / num_steps, jnp.sqrt(2) - 1), 0.0)
    gammas = jnp.where(s_min <= sigmas, gammas, 0.0)
    return gammas

  return gamma


def get_times(num_steps=1000, dt=None, t0=None):
  """
  Get linear, monotonically increasing time schedule.
  Args:
      num_steps: number of discretization time steps.
      dt: time step duration, float or `None`.
        Optional, if provided then final time, t1 = dt * num_steps.
      t0: A small float 0. < t0 << 1. The SDE or ODE are integrated to
          t0 to avoid numerical issues.
  Return:
      ts: JAX array of monotonically increasing values t \in [t0, t1].
  """
  if dt is not None:
    if t0 is not None:
      t1 = dt * (num_steps - 1) + t0
      # Defined in forward time, t \in [t0, t1], 0 < t0 << t1
      ts, step = jnp.linspace(t0, t1, num_steps, retstep=True)
      ts = ts.reshape(-1, 1)
      assert jnp.isclose(step, (t1 - t0) / (num_steps - 1))
      assert jnp.isclose(step, dt)
      dt = step
      assert t0 == ts[0]
    else:
      t1 = dt * num_steps
      # Defined in forward time, t \in [dt , t1], 0 < \t0 << t1
      ts, step = jnp.linspace(0.0, t1, num_steps + 1, retstep=True)
      ts = ts[1:].reshape(-1, 1)
      assert jnp.isclose(step, dt)
      dt = step
      t0 = ts[0]
  else:
    t1 = 1.0
    if t0 is not None:
      ts, dt = jnp.linspace(t0, 1.0, num_steps, retstep=True)
      ts = ts.reshape(-1, 1)
      assert jnp.isclose(dt, (1.0 - t0) / (num_steps - 1))
      assert t0 == ts[0]
    else:
      # Defined in forward time, t \in [dt, 1.0], 0 < dt << 1
      ts, dt = jnp.linspace(0.0, 1.0, num_steps + 1, retstep=True)
      ts = ts[1:].reshape(-1, 1)
      assert jnp.isclose(dt, 1.0 / num_steps)
      t0 = ts[0]
  assert ts[0, 0] == t0
  assert ts[-1, 0] == t1
  dts = jnp.diff(ts)
  assert jnp.all(dts > 0.0)
  assert jnp.all(dts == dt)
  return ts, dt


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


def batch_matmul_A(A, b):
  return vmap(lambda b: A @ b)(b)


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
    return batch_mul(noise, 1.0 / std) + score(x, t)


def get_pointwise_loss(
  sde,
  model,
  score_scaling=True,
  likelihood_weighting=True,
  reduce_mean=True,
):
  """Create a loss function for score matching training, returning a function that can evaluate the loss pointwise over time.
  Args:
    sde: Instantiation of a valid SDE class.
    solver: Instantiation of a valid Solver class.
    model: A valid flax neural network `:class:flax.linen.Module` class.
    score_scaling: Bool, set to `True` if learning a score scaled by the marginal standard deviation.
    likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    reduce_mean: Bool, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.

  Returns:
    A loss function that can be used for score matching training and can evaluate the loss pointwise over time.
  """
  reduce_op = (
    jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  )

  def pointwise_loss(t, params, rng, data):
    n_batch = data.shape[0]
    ts = jnp.ones((n_batch,)) * t
    score = get_score(sde, model, params, score_scaling)
    e = errors(ts, sde, score, rng, data, likelihood_weighting)
    losses = e**2
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    if likelihood_weighting:
      g2 = sde.sde(jnp.zeros_like(data), ts)[1] ** 2
      losses = losses * g2
    return jnp.mean(losses)

  return pointwise_loss


def get_loss(
  sde,
  solver,
  model,
  score_scaling=True,
  likelihood_weighting=True,
  reduce_mean=True,
):
  """Create a loss function for score matching training.
  Args:
    sde: Instantiation of a valid SDE class.
    solver: Instantiation of a valid Solver class.
    model: A valid flax neural network `:class:flax.linen.Module` class.
    score_scaling: Bool, set to `True` if learning a score scaled by the marginal standard deviation.
    likelihood_weighting: Bool, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    reduce_mean: Bool, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.

  Returns:
    A loss function that can be used for score matching training and is an expectation of the regression loss over time.
  """
  reduce_op = (
    jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  )

  def loss(params, rng, data):
    rng, step_rng = random.split(rng)
    ts = random.uniform(
      step_rng, (data.shape[0],), minval=solver.ts[0], maxval=solver.t1
    )
    score = get_score(sde, model, params, score_scaling)
    e = errors(ts, sde, score, rng, data, likelihood_weighting)
    losses = e**2
    losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    if likelihood_weighting:
      g2 = sde.sde(jnp.zeros_like(data), ts)[1] ** 2
      losses = losses * g2
    return jnp.mean(losses)

  return loss


class EDM2Loss:
  """
  Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
  paper "Analyzing and Improving the Training Dynamics of Diffusion Models".
  """

  def __init__(
    self, net, batch_gpu_total, loss_scaling=1.0, p_mean=-0.4, p_std=1.0, sigma_data=0.5
  ):
    self.net = net
    self.p_mean = p_mean
    self.p_std = p_std
    self.sigma_data = sigma_data
    self.loss_scaling = loss_scaling
    self.batch_gpu_total = batch_gpu_total

  def __call__(self, params, rng, data, labels=None):
    rng, step_rng = random.split(rng)
    random_normal = random.normal(
      step_rng, (data.shape[0],) + (1,) * (len(data.shape) - 1)
    )
    print("r", random_normal.shape)
    sigma = jnp.exp(random_normal * self.p_std + self.p_mean)
    print("rs",sigma.shape)
    weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
    noise = random.normal(step_rng, data.shape) * sigma
    denoised, logvar = self.net.apply(params, data + noise, sigma, labels)
    loss = (weight / jnp.exp(logvar)) * ((denoised - data) ** 2) + logvar
    return jnp.sum(loss) * (self.loss_scaling / self.batch_gpu_total)


def get_score(sde, model, params, score_scaling):
  if score_scaling is True:
    return lambda x, t: -batch_mul(
      model.apply(params, x, t), 1.0 / jnp.sqrt(sde.variance(t))
    )
  else:
    return lambda x, t: -model.apply(params, x, t)


def get_net(model, params):
  # TODO: compare to edmv2 code and work out if it is correct
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


def get_sampler(
  shape,
  outer_solver,
  inner_solver=None,
  denoise=True,
  stack_samples=False,
  inverse_scaler=None,
):
  """Get a sampler from (possibly interleaved) numerical solver(s).

  Args:
    shape: Shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
      of the object being sampled from, for example, an image may have
      x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
    outer_solver: A valid numerical solver class that will act on an outer loop.
    inner_solver: '' that will act on an inner loop.
    denoise: Bool, that if `True` applies one-step denoising to final samples.
    stack_samples: Bool, that if `True` return all of the sample path or
      just returns the last sample.
    inverse_scaler: The inverse data normalizer function.
  Returns:
    A sampler.
  """
  if inverse_scaler is None:
    inverse_scaler = lambda x: x

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
    outer_update = partial(shared_update, solver=outer_solver)
    outer_ts = outer_solver.ts

    if inner_solver:
      inner_update = partial(shared_update, solver=inner_solver)
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
        (rng, x, x_mean, vec_t), _ = scan(
          inner_step, (step_rng, x, x_mean, vec_t), inner_ts
        )
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
      if inner_solver:
        x = inner_solver.prior(step_rng, shape)
      else:
        x = outer_solver.prior(step_rng, shape)
    else:
      assert x_0.shape == shape
      x = x_0
    if not stack_samples:
      (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
      return inverse_scaler(x_mean if denoise else x), num_function_evaluations
    else:
      (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
      return inverse_scaler(xs), num_function_evaluations

  # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
  return sampler
