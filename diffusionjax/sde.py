"""SDE class."""

import jax.numpy as jnp
from jax import random, vmap
from diffusionjax.utils import (
  batch_mul,
  get_exponential_sigma_function,
  get_linear_beta_function,
)


def ulangevin(score, x, t):
  drift = -score(x, t)
  diffusion = jnp.ones(x.shape) * jnp.sqrt(2)
  return drift, diffusion


class ULangevin:
  """Unadjusted Langevin SDE."""

  def __init__(self, score):
    self.score = score
    self.sde = lambda x, t: ulangevin(self.score, x, t)


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


class VE:
  """Variance exploding (VE) SDE, a.k.a. diffusion process with a time dependent diffusion coefficient."""

  def __init__(self, sigma=None):
    if sigma is None:
      self.sigma = get_exponential_sigma_function(sigma_min=0.01, sigma_max=378.0)
    else:
      self.sigma = sigma
    self.sigma_min = self.sigma(0.0)
    self.sigma_max = self.sigma(1.0)

  def sde(self, x, t):
    sigma_t = self.sigma(t)
    drift = jnp.zeros_like(x)
    diffusion = sigma_t * jnp.sqrt(
      2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min))
    )

    return drift, diffusion

  def mean_coeff(self, t):
    return jnp.ones_like(t)

  def variance(self, t):
    return self.sigma(t) ** 2

  def prior(self, rng, shape):
    return random.normal(rng, shape) * self.sigma_max

  def reverse(self, score):
    forward_sde = self.sde
    sigma = self.sigma

    return RVE(score, forward_sde, sigma)

  def r2(self, t, data_variance):
    r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
    the data distribution is isotropic-Gaussian.

    .. math::
      \text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
      \text{ and } p_{t|0}(x_{t}|x_{0}) = \mathcal{N}(x_0, \sigma_{t}^{2}I)
    """
    variance = self.variance(t)
    return variance * data_variance / (variance + data_variance)

  def ratio(self, t):
    """Ratio of marginal variance and mean coeff."""
    return self.variance(t)


class VP:
  """Variance preserving (VP) SDE, a.k.a. time rescaled Ohrnstein Uhlenbeck (OU) SDE."""

  def __init__(self, beta=None, mean_coeff=None):
    if beta is None:
      self.beta, self.mean_coeff = get_linear_beta_function(
        beta_min=0.1, beta_max=20.0
      )
    else:
      self.beta = beta
      self.mean_coeff = mean_coeff

  def sde(self, x, t):
    beta_t = self.beta(t)
    drift = -0.5 * batch_mul(beta_t, x)
    diffusion = jnp.sqrt(beta_t)
    return drift, diffusion

  def std(self, t):
    return jnp.sqrt(self.variance(t))

  def variance(self, t):
    return 1.0 - self.mean_coeff(t)**2

  def marginal_prob(self, x, t):
    return batch_mul(self.mean_coeff(t), x), jnp.sqrt(self.variance(t))

  def prior(self, rng, shape):
    return random.normal(rng, shape)

  def reverse(self, score):
    fwd_sde = self.sde
    beta = self.beta
    mean_coeff = self.mean_coeff
    return RVP(score, fwd_sde, beta, mean_coeff)

  def r2(self, t, data_variance):
    r"""Analytic variance of the distribution at time zero conditioned on x_t, given crude assumption that
    the data distribution is isotropic-Gaussian.

    .. math::
      \text{Variance of }p_{0}(x_{0}|x_{t}) \text{ if } p_{0}(x_{0}) = \mathcal{N}(0, \text{data_variance}I)
      \text{ and } p_{t|0}(x_{t}|x_{0}) = \mathcal{N}(\sqrt(\alpha_{t})x_0, (1 - \alpha_{t})I)
    """
    alpha = self.mean_coeff(t)**2
    variance = 1.0 - alpha
    return variance * data_variance / (variance + alpha * data_variance)

  def ratio(self, t):
    """Ratio of marginal variance and mean coeff."""
    return self.variance(t) / self.mean_coeff(t)


class RVE(RSDE, VE):
  def get_estimate_x_0_vmap(self, observation_map):
    """
    Get a function returning the MMSE estimate of x_0|x_t.

    Args:
      observation_map: function that operates on unbatched x.
      shape: optional tuple that reshapes x so that it can be operated on.
    """

    def estimate_x_0(x, t):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = x + v_t * s
      return observation_map(x_0), (s, x_0)

    return estimate_x_0

  def get_estimate_x_0(self, observation_map, shape=None):
    """
    Get a function returning the MMSE estimate of x_0|x_t.

    Args:
      observation_map: function that operates on unbatched x.
      shape: optional tuple that reshapes x so that it can be operated on.
    """
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t):
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = x + batch_mul(v_t, s)
      if shape:
        return batch_observation_map(x_0.reshape(shape)), (s, x_0)
      else:
        return batch_observation_map(x_0), (s, x_0)

    return estimate_x_0

  def guide(self, get_guidance_score, observation_map, *args, **kwargs):
    guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
    return RVE(guidance_score, self.forward_sde, self.sigma)

  def correct(self, corrector):
    class CVE(RVE):
      def sde(x, t):
        return corrector(self.score, x, t)

    return CVE(self.score, self.forward_sde, self.sigma)


class RVP(RSDE, VP):
  def get_estimate_x_0_vmap(self, observation_map):
    """
    Get a function returning the MMSE estimate of x_0|x_t.

    Args:
      observation_map: function that operates on unbatched x.
      shape: optional tuple that reshapes x so that it can be operated on.
    """

    def estimate_x_0(x, t):
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      m_t = self.mean_coeff(t)
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = (x + v_t * s) / m_t
      return observation_map(x_0), (s, x_0)

    return estimate_x_0

  def get_estimate_x_0(self, observation_map, shape=None):
    """
    Get a function returning the MMSE estimate of x_0|x_t.

    Args:
      observation_map: function that operates on unbatched x.
      shape: optional tuple that reshapes x so that it can be operated on.
    """
    batch_observation_map = vmap(observation_map)

    def estimate_x_0(x, t):
      m_t = self.mean_coeff(t)
      v_t = self.variance(t)
      s = self.score(x, t)
      x_0 = batch_mul(x + batch_mul(v_t, s), 1.0 / m_t)
      if shape:
        return batch_observation_map(x_0.reshape(shape)), (s, x_0)
      else:
        return batch_observation_map(x_0), (s, x_0)

    return estimate_x_0

  def guide(self, get_guidance_score, observation_map, *args, **kwargs):
    guidance_score = get_guidance_score(self, observation_map, *args, **kwargs)
    return RVP(guidance_score, self.forward_sde, self.beta, self.mean_coeff)

  def correct(self, corrector):
    class CVP(RVP):
      def sde(x, t):
        return corrector(self.score, x, t)

    return CVP(self.score, self.forward_sde, self.beta, self.mean_coeff)
