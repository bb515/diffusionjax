"""Solver classes, including Markov chains."""
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from diffusionjax.utils import (
  batch_mul, batch_mul_A, get_times, get_timestep,
  get_sigma_function, get_linear_beta_function, continuous_to_discrete)
import abc


class Solver(abc.ABC):
  """SDE solver abstract class. Functions are designed for a mini-batch of inputs."""
  def __init__(self, ts=None):
    """Construct a Solver. Note that for continuous time we choose to control for numerical
    error by using a beta schedule instead of an adaptive time step schedule, since adaptive
    time steps are equivalent to a beta schedule, and beta schedule hyperparameters have
    been explored extensively in the literature. Therefore, the timesteps must be equally
    spaced by dt.
    Args:
      ts: JAX array of equally spaced, monotonically increasing values t in [t0, t1].
    """
    if ts is None: ts, _ = get_times(num_steps=1000)
    self.ts = ts
    self.t1 = ts[-1]
    self.t0 = ts[0]
    self.dt = ts[1] - ts[0]
    self.num_steps = ts.size
    # print("t \in [{}, {}] with step size dt={} and num_steps={}".format(
    #   self.t0, self.t1, self.dt, self.num_steps))
    dts = jnp.diff(ts)
    if not jnp.all(dts > 0.):
      raise ValueError("ts must be monotonically increasing, got ts={}".format(ts))
    if not jnp.all(dts == self.dt):
      raise ValueError("stepsize dt must be constant; ts must be equally \
      spaced, got diff(ts)={}".format(dts))


  @abc.abstractmethod
  def update(self, rng, x, t):
    """Return the update of the state and any auxilliary values.

    Args:
      rng: A JAX random state.
      x: A JAX array of the state.
      t: JAX array of the time.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """


class EulerMaruyama(Solver):
  """Euler Maruyama numerical solver of an SDE.
  Functions are designed for a mini-batch of inputs."""

  def __init__(self, sde, ts=None):
    """Constructs an Euler-Maruyama Solver.
    Args:
      sde: A valid SDE class.
    """
    super().__init__(ts)
    self.sde = sde
    self.prior = sde.prior

  def update(self, rng, x, t):
    drift, diffusion = self.sde.sde(x, t)
    f = drift * self.dt
    G = diffusion * jnp.sqrt(self.dt)
    noise = random.normal(rng, x.shape)
    x_mean = x + f
    x = x_mean + batch_mul(G, noise)
    return x, x_mean


class Annealed(Solver):
  """Annealed Langevin numerical solver of an SDE.
  Functions are designed for a mini-batch of inputs.
  Sampler must be `pmap` over "batch" axis as
  suggested by https://arxiv.org/abs/2011.13456 Song
  et al.
  """

  def __init__(self, sde, snr=1e-2, ts=jnp.empty((2, 1))):
    """Constructs an Annealed Langevin Solver.
    Args:
      sde: A valid SDE class.
      snr: A hyperparameter representing a signal-to-noise ratio.
      ts: For a corrector, just need a placeholder JAX array with length
        number of timesteps of the inner solver.
    """
    super().__init__(ts)
    self.sde = sde
    self.snr = snr
    self.prior = sde.prior

  def update(self, rng, x, t):
    grad = self.sde.score(x, t)
    grad_norm = jnp.linalg.norm(
      grad.reshape((grad.shape[0], -1)), axis=-1).mean()
    grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
    noise = random.normal(rng, x.shape)
    noise_norm = jnp.linalg.norm(
      noise.reshape((noise.shape[0], -1)), axis=-1).mean()
    noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
    alpha = jnp.exp(2 * self.sde.log_mean_coeff(t))
    dt = (self.snr * noise_norm / grad_norm)**2 * 2 * alpha
    x_mean = x + batch_mul(grad, dt)
    x = x_mean + batch_mul(2 * dt, noise)
    return x, x_mean


class Inpainted(Solver):
  """Inpainting constraint for numerical solver of an SDE.
  Functions are designed for a mini-batch of inputs."""

  def __init__(self, sde, mask, y, ts=jnp.empty((1, 1))):
    """Constructs an Annealed Langevin Solver.
    Args:
      sde: A valid SDE class.
      snr: A hyperparameter representing a signal-to-noise ratio.
    """
    super().__init__(ts)
    self.sde = sde
    self.mask = mask
    self.y = y

  def prior(self, rng, shape):
    x = self.sde.prior(rng, shape)
    x = batch_mul_A((1. - self.mask), x) + self.y * self.mask
    return x

  def update(self, rng, x, t):
    mean_coeff = self.sde.mean_coeff(t)
    std = jnp.sqrt(self.sde.variance(t))
    masked_data_mean = batch_mul_A(self.y, mean_coeff)
    masked_data = masked_data_mean + batch_mul(random.normal(rng, x.shape), std)
    x = batch_mul_A((1. - self.mask), x) + batch_mul_A(self.mask, masked_data)
    x_mean = batch_mul_A((1. - self.mask), x) + batch_mul_A(self.mask, masked_data_mean)
    return x, x_mean


class Projected(Solver):
  """Inpainting constraint for numerical solver of an SDE.
  Functions are designed for a mini-batch of inputs."""

  def __init__(self, sde, mask, y, coeff=1., ts=jnp.empty((1, 1))):
    """Constructs an Annealed Langevin Solver.
    Args:
      sde: A valid SDE class.
      snr: A hyperparameter representing a signal-to-noise ratio.
    """
    super().__init__(ts)
    self.sde = sde
    self.mask = mask
    self.y = y
    self.coeff = coeff
    self.prior = sde.prior

  def merge_data_with_mask(self, x_space, data, mask, coeff):
    return batch_mul_A(mask * coeff, data) + batch_mul_A((1. - mask * coeff), x_space)

  def update(self, rng, x, t):
    mean_coeff = self.sde.mean_coeff(t)
    masked_data_mean = batch_mul_A(self.y, mean_coeff)
    std = jnp.sqrt(self.sde.variance(t))
    z_data = masked_data_mean + batch_mul(std, random.normal(rng, x.shape))
    x = self.merge_data_with_mask(x, z_data, self.mask, self.coeff)
    x_mean = self.merge_data_with_mask(x, masked_data_mean, self.mask, self.coeff)
    return x, x_mean


class DDPM(Solver):
  """DDPM Markov chain using Ancestral sampling."""
  def __init__(self, score, beta=None, ts=None):
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=.1, beta_max=20.)
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.score = score
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
    self.alphas_cumprod_prev = jnp.append(1., self.alphas_cumprod[:-1])
    self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
    self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1. - self.alphas_cumprod_prev)

  def get_estimate_x_0_vmap(self, observation_map):

    def estimate_x_0(x, t, timestep):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      m = self.sqrt_alphas_cumprod[timestep]
      v = self.sqrt_1m_alphas_cumprod[timestep]**2
      s = self.score(x, t)
      x_0 = (x + v * s) / m
      return observation_map(x_0), (s, x_0)
    return estimate_x_0

  def get_estimate_x_0(self, observation_map):
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t, timestep):
      m = self.sqrt_alphas_cumprod[timestep]
      v = self.sqrt_1m_alphas_cumprod[timestep]**2
      s = self.score(x, t)
      x_0 = batch_mul(x + batch_mul(v, s), 1. / m)
      return batch_observation_map(x_0), (s, x_0)
    return estimate_x_0

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def posterior(self, score, x, timestep):
    beta = self.discrete_betas[timestep]
    # As implemented by Song
    # https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sampling.py#L237C5-L237C79
    # x_mean = batch_mul(
    #     (x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))  # DDPM
    # std = jnp.sqrt(beta)

    # # As implemented by DPS2022
    # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/gaussian_diffusion.py#L373
    m = self.sqrt_alphas_cumprod[timestep]
    v = self.sqrt_1m_alphas_cumprod[timestep]**2
    alpha = self.alphas[timestep]
    x_0 = batch_mul((x + batch_mul(v, score)), 1. / m)
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    x_mean = batch_mul(jnp.sqrt(alpha) * v_prev / v, x) + batch_mul(m_prev * beta / v, x_0)
    std = jnp.sqrt(beta * v_prev / v)
    return x_mean, std

  def update(self, rng, x, t):
    score = self.score(x, t)
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    x_mean, std = self.posterior(score, x, timestep)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class SMLD(Solver):
  """SMLD(NCSN) Markov Chain using Ancestral sampling."""
  def __init__(self, score, sigma=None, ts=None):
    super().__init__(ts)
    if sigma is None:
      sigma = get_sigma_function(sigma_min=0.01, sigma_max=378.)
    sigmas = vmap(sigma)(ts.flatten())
    self.sigma_max = sigmas[-1]
    self.discrete_sigmas = sigmas
    self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])
    self.score = score

  def get_estimate_x_0_vmap(self, observation_map):

    def estimate_x_0(x, t, timestep):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      v = self.discrete_sigmas[timestep]**2
      s = self.score(x, t)
      x_0 = x + v * s
      return observation_map(x_0), (s, x_0)
    return estimate_x_0

  def get_estimate_x_0(self, observation_map):
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t, timestep):
      v = self.discrete_sigmas[timestep]**2
      s = self.score(x, t)
      x_0 = x + batch_mul(v, s)
      return batch_observation_map(x_0), (s, x_0)
    return estimate_x_0

  def prior(self, rng, shape):
    return random.normal(rng, shape) * self.sigma_max

  def posterior(self, score, x, timestep):
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]

    # As implemented by Song https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sampling.py#L220
    # x_mean = x + batch_mul(score, sigma**2 - sigma_prev**2)
    # std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))

    # From posterior in Appendix F https://arxiv.org/pdf/2011.13456.pdf
    x_0 = x + batch_mul(sigma**2, score)
    x_mean = batch_mul(sigma_prev**2 / sigma**2, x) + batch_mul(1 - sigma_prev**2 / sigma**2, x_0)
    std = jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    return x_mean, std

  def update(self, rng, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    score = self.score(x, t)
    x_mean, std = self.posterior(score, x, timestep)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class DDIMVP(Solver):
  """DDIM Markov chain. For the DDPM Markov Chain or VP SDE."""

  def __init__(self, model, eta=0., beta=None, ts=None):
    """
    Args:
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.` is DDPMVP.
    """
    super().__init__(ts)
    if beta is None:
      beta, _ = get_linear_beta_function(
        beta_min=.1, beta_max=20.)
    self.discrete_betas = continuous_to_discrete(vmap(beta)(self.ts.flatten()), self.dt)
    self.eta = eta
    self.model = model
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
    self.alphas_cumprod_prev = jnp.append(1., self.alphas_cumprod[:-1])
    self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
    self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1. - self.alphas_cumprod_prev)

  def get_estimate_x_0_vmap(self, observation_map):

    def estimate_x_0(x, t, timestep):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      m = self.sqrt_alphas_cumprod[timestep]
      sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
      epsilon = self.model(x, t)
      x_0 = (x - sqrt_1m_alpha * epsilon) / m
      return observation_map(x_0), (epsilon, x_0)
    return estimate_x_0

  def get_estimate_x_0(self, observation_map):
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t, timestep):
      m = self.sqrt_alphas_cumprod[timestep]
      sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
      epsilon = self.model(x, t)
      x_0 = batch_mul(x - batch_mul(sqrt_1m_alpha, epsilon), 1. / m)
      return batch_observation_map(x_0), (epsilon, x_0)
    return estimate_x_0

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def posterior(self, x, t):
    # # As implemented by DPS2022
    # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/gaussian_diffusion.py#L373
    # and as written in https://arxiv.org/pdf/2010.02502.pdf
    epsilon = self.model(x, t)
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha_cumprod = self.alphas_cumprod[timestep]
    alpha_cumprod_prev = self.alphas_cumprod_prev[timestep]
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1. / m)
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha_cumprod / alpha_cumprod_prev))
    coeff2 = jnp.sqrt(v_prev - coeff1**2)
    x_mean = batch_mul(m_prev, x_0) + batch_mul(coeff2, epsilon)
    std = coeff1
    return x_mean, std

  def update(self, rng, x, t):
    x_mean, std = self.posterior(x, t)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class DDIMVE(Solver):
  """DDIM Markov chain. For the SMLD Markov Chain or VE SDE.
    Args:
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.` is DDPMVE.
    """

  def __init__(self, model, eta=0., sigma=None, ts=None):
    super().__init__(ts)
    if sigma is None:
      sigma = get_sigma_function(sigma_min=0.01, sigma_max=378.)
    sigmas = vmap(sigma)(ts.flatten())
    self.sigma_max = sigmas[-1]
    self.discrete_sigmas = sigmas
    self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])
    self.eta = eta
    self.model = model

  def get_estimate_x_0_vmap(self, observation_map):

    def estimate_x_0(x, t, timestep):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      std = self.discrete_sigmas[timestep]
      epsilon = self.model(x, t)
      x_0 = x - std * epsilon
      return observation_map(x_0), (epsilon, x_0)
    return estimate_x_0

  def get_estimate_x_0(self, observation_map):
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t, timestep):
      std = self.discrete_sigmas[timestep]
      epsilon = self.model(x, t)
      x_0 = x - batch_mul(std, epsilon)
      return batch_observation_map(x_0), (epsilon, x_0)
    return estimate_x_0

  def prior(self, rng, shape):
    return random.normal(rng, shape) * self.sigma_max

  def posterior(self, x, t):
    timestep = get_timestep(t, self.t0, self.t1, self.num_steps)
    epsilon = self.model(x, t)
    sigma = self.discrete_sigmas[timestep]
    sigma_prev = self.discrete_sigmas_prev[timestep]
    coeff1 = self.eta * jnp.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
    coeff2 = jnp.sqrt(sigma_prev**2  - coeff1**2)

    # Eq.(18) Appendix A.4 https://openreview.net/pdf/210093330709030207aa90dbfe2a1f525ac5fb7d.pdf
    x_0 = x - batch_mul(sigma, epsilon)
    x_mean = x_0 + batch_mul(coeff2, epsilon)

    # Eq.(19) Appendix A.4 https://openreview.net/pdf/210093330709030207aa90dbfe2a1f525ac5fb7d.pdf
    # score = - batch_mul(1. / sigma, epsilon)
    # x_mean = x + batch_mul(sigma * (sigma - coeff2), score)

    std = coeff1
    return x_mean, std

  def update(self, rng, x, t):
    x_mean, std = self.posterior(x, t)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean
