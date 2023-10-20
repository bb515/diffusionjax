"""SDE class."""
import jax.numpy as jnp
from jax import random, vmap
from diffusionjax.utils import batch_mul


def udlangevin(score, x, t):
  drift = -score(x, t)
  diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
  return drift, diffusion


class RSDE:
  """Reverse SDE class."""
  def __init__(self, score, forward_sde, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.score = score
    self.forward_sde = forward_sde

  def sde(self, x, t):
    drift, diffusion = self.forward_sde(x, t)
    drift = -drift + batch_mul(diffusion**2, self.score(x, t))
    return drift, diffusion


class ODLangevin:
  """Overdamped langevin SDE."""
  def __init__(self, score, damping=2e0, L=1.0):
    self.score = score
    self.damping = damping
    self.L = L

  def sde(self, x, t):
    drift = -self.score(x, t)
    diffusion = jnp.ones(x.shape) * jnp.sqrt(2 * self.damping / self.L)
    return drift, diffusion


class UDLangevin:
  """Underdamped Langevin SDE."""
  def __init__(self, score):
    self.score = score
    self.sde = lambda x, t: udlangevin(self.score, x, t)


class VE:
  """Variance exploding (VE) SDE, a.k.a. diffusion process with a time dependent diffusion coefficient."""
  def __init__(self, sigma_min=0.01, sigma_max=378.):
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min)**t
    drift = jnp.zeros_like(x)
    diffusion = sigma * jnp.sqrt(2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min)))
    return drift, diffusion

  def log_mean_coeff(self, t):
    return jnp.zeros_like(t)

  def mean_coeff(self, t):
    return jnp.ones_like(t)

  def variance(self, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min)**t
    return std**2

  def prior(self, rng, shape):
    return random.normal(rng, shape) * self.sigma_max

  def reverse(self, score):
    forward_sde = self.sde
    sigma_min = self.sigma_min
    sigma_max = self.sigma_max

    return RVE(score, forward_sde, sigma_min, sigma_max)

  def r2(self, t, data_variance):
    r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
    the data distribution is isotropic-Gaussian.

    .. math::
      \text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
    """
    variance = self.variance(t)
    return variance * data_variance / (variance + data_variance)

  def ratio(self, t):
    """Ratio of marginal variance and mean coeff."""
    return self.variance(t)


class VP:
  """Variance preserving (VP) SDE, a.k.a. time rescaled Ohrnstein Uhlenbeck (OU) SDE."""
  def __init__(self, beta_min=0.1, beta_max=20.):
    self.beta_min = beta_min
    self.beta_max = beta_max

  def sde(self, x, t):
    beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
    drift = -0.5 * batch_mul(beta_t, x)
    diffusion = jnp.sqrt(beta_t)
    return drift, diffusion

  def log_mean_coeff(self, t):
    return -0.5 * t * self.beta_min - 0.25 * t**2 * (self.beta_max - self.beta_min)

  def mean_coeff(self, t):
    return jnp.exp(self.log_mean_coeff(t))

  def variance(self, t):
    return 1.0 - jnp.exp(2 * self.log_mean_coeff(t))

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def reverse(self, score):
    fwd_sde = self.sde
    beta_min = self.beta_min
    beta_max = self.beta_max
    return RVP(score, fwd_sde, beta_min, beta_max)

  def r2(self, t, data_variance):
    r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
    the data distribution is isotropic-Gaussian.

    .. math::
      \text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
    """
    alpha = jnp.exp(2 * self.log_mean_coeff(t))
    return (1 - alpha) * data_variance / (1 - alpha + alpha * data_variance)

  def ratio(self, t):
    """Ratio of marginal variance and mean coeff."""
    return self.variance(t) / self.mean_coeff(t)


class RVE(RSDE, VE):

  def get_estimate_x_0(self, observation_map):
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(self, x, t):
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = x + v_t * s
      return batch_observation_map(x_0), (s, x_0)
    return estimate_x_0

  def guide(self, get_guidance_score, observation_map, *args, **kwargs):
    guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
    return RVE(guidance_score, self.forward_sde, self.sigma_min, self.sigma_max)

  def correct(self, corrector):

    class CVE(RVE):

      def sde(x, t):
        return corrector(self.score, x, t)

    return CVE(self.score, self.forward_sde, self.sigma_min, self.sigma_max)


class RVP(RSDE, VP):

  def get_estimate_x_0(self, observation_map):
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t):
      m_t = self.mean_coeff(t)
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = batch_mul(x + batch_mul(v_t, s), 1. / m_t)
      return batch_observation_map(x_0), (s, x_0)
    return estimate_x_0

  def guide(self, get_guidance_score, observation_map, *args, **kwargs):
    guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
    return RVP(guidance_score, self.forward_sde, self.beta_min, self.beta_max)

  def correct(self, corrector):

    class CVP(RVP):

      def sde(x, t):
        return corrector(self.score, x, t)

    return CVP(self.score, self.forward_sde, self.beta_min, self.beta_max)
