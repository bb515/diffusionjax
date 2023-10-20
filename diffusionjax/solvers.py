"""Solver classes, including Markov chains."""
import jax
import jax.numpy as jnp
import jax.random as random
from jax import vmap
from diffusionjax.utils import batch_mul
import abc


class Solver(abc.ABC):
  """Solver abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, num_steps=1000, dt=None, epsilon=None):
    """Construct a Solver.
    Args:
      num_steps: number of discretization time steps.
      dt: time step duration, float or `None`.
        Optional, if provided then final time, t1 = dt * num_steps.
      epsilon: A small float 0. < `epsilon` << 1. The SDE or ODE are integrated to `epsilon` to avoid numerical issues.
    """
    self.num_steps = num_steps

    # Handle four different ways of specifying the discretisation step size, its terminal time and its total time.
    if dt is not None:
      self.t1 = dt * num_steps
      if epsilon is not None:
        # Defined in forward time, t \in [epsilon, t1], 0 < epsilon << t1
        ts, step = jnp.linspace(epsilon, self.t1, num_steps, retstep=True)
        self.ts = ts.reshape(-1, 1)
        assert step == (self.t1 - epsilon) / num_steps
        self.dt = step
      else:
        # Defined in forward time, t \in [dt , t1], 0 < \epsilon << t1
        ts, step = jnp.linspace(0, self.t1, num_steps + 1, retstep=True)
        self.ts = ts[1:].reshape(-1, 1)
        assert step == dt
        self.dt = step
    else:
      self.t1 = 1.0
      if epsilon is not None:
        self.ts, step = jnp.linspace(epsilon, 1, num_steps, retstep=True)
        assert step == (1. - epsilon) / num_steps
        self.dt = step
      else:
        # Defined in forward time, t \in [dt, 1.0], 0 < dt << 1
        ts, step = jnp.linspace(0, 1, num_steps + 1, retstep=True)
        self.ts = ts[1:].reshape(-1, 1)
        assert step == 1. / num_steps
        self.dt = step

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

  def __init__(self, sde, num_steps=1000, dt=None, epsilon=None):
    """Constructs an Euler-Maruyama Solver.
    Args:
      sde: A valid SDE class.
    """
    super().__init__(num_steps=num_steps, dt=dt, epsilon=epsilon)
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
  Functions are designed for a mini-batch of inputs."""

  def __init__(self, sde, snr=1e-2, num_steps=2, dt=None, epsilon=None):
    """Constructs an Annealed Langevin Solver.
    Args:
      sde: A valid SDE class.
      snr: A hyperparameter representing a signal-to-noise ratio.
    """
    super().__init__(num_steps, dt=dt, epsilon=epsilon)
    self.sde = sde
    self.snr = snr
    # self.prior = sde.prior

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


class DDPM(Solver):
  """DDPM Markov chain using Ancestral sampling."""
  def __init__(self, score, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
    super().__init__(num_steps, dt, epsilon)
    self.score = score
    self.discrete_betas = jnp.linspace(
      beta_min / num_steps, beta_max / num_steps, num_steps)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
    self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
    self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
    self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1. - self.alphas_cumprod_prev)

  def get_estimate_x_0_vmap(self, shape, observation_map):

    def estimate_x_0(x, t, timestep):
      m = self.sqrt_alphas_cumprod[timestep]
      v = self.sqrt_1m_alphas_cumprod[timestep]**2
      x = x.reshape(shape[1:])
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      s = self.score(x, t)
      s = s.flatten()
      x = x.flatten()
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
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    x_mean, std = self.posterior(score, x, timestep)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class SMLD(Solver):
  """SMLD(NCSN) Markov Chain using Ancestral sampling."""

  def __init__(self, score, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
    super().__init__(num_steps, dt, epsilon)
    self.score = score
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = jnp.exp(
        jnp.linspace(jnp.log(self.sigma_min),
                     jnp.log(self.sigma_max),
                     self.num_steps))
    self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])

  def get_estimate_x_0_vmap(self, shape, observation_map):

    def estimate_x_0(x, t, timestep):
      v = self.discrete_sigmas[timestep]**2
      x = x.reshape(shape)
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      s = self.score(x, t)
      x = x.flatten()
      s = s.flatten()
      x_0 = x + v * s
      return observation_map(x_0), (s, x_0)
    return estimate_x_0

  def get_estimate_x_0(self, observation_map):

    def estimate_x_0(x, t, timestep):
      v = self.discrete_sigmas[timestep]**2
      s = self.score(x, t)
      x_0 = x + batch_mul(v, s)
      return observation_map(x_0), (s, x_0)
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
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    score = self.score(x, t)
    x_mean, std = self.posterior(score, x, timestep)
    z = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, z)
    return x, x_mean


class DDIMVP(Solver):
  """DDIM Markov chain. For the DDPM Markov Chain or VP SDE."""

  def __init__(self, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, beta_min=0.1, beta_max=20.0):
    """
    Args:
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVP.
    """
    super().__init__(num_steps, dt, epsilon)
    self.eta = eta
    self.model = model
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.discrete_betas = jnp.linspace(
      beta_min / num_steps, beta_max / num_steps, num_steps)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
    self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
    self.sqrt_alphas_cumprod_prev = jnp.sqrt(self.alphas_cumprod_prev)
    self.sqrt_1m_alphas_cumprod_prev = jnp.sqrt(1. - self.alphas_cumprod_prev)

  def get_estimate_x_0_vmap(self, shape, observation_map):

    def estimate_x_0(x, t, timestep):
      m = self.sqrt_alphas_cumprod[timestep]
      sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
      # x = x.reshape(shape)
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      epsilon = self.model(x, t)
      epsilon = epsilon.flatten()
      x = x.flatten()
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
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
    m = self.sqrt_alphas_cumprod[timestep]
    sqrt_1m_alpha = self.sqrt_1m_alphas_cumprod[timestep]
    v = sqrt_1m_alpha**2
    alpha = m**2
    m_prev = self.sqrt_alphas_cumprod_prev[timestep]
    v_prev = self.sqrt_1m_alphas_cumprod_prev[timestep]**2
    alpha_prev = m_prev**2
    x_0 = batch_mul((x - batch_mul(sqrt_1m_alpha, epsilon)), 1. / m)
    coeff1 = self.eta * jnp.sqrt((v_prev / v) * (1 - alpha / alpha_prev))
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
  """DDIM Markov chain. For the SMLD Markov Chain or VE SDE."""

  def __init__(self, model, eta=0.0, num_steps=1000, dt=None, epsilon=None, sigma_min=0.01, sigma_max=378.):
    """
    Args:
        model: DDIM parameterizes the `epsilon(x, t) = -1. * fwd_marginal_std(t) * score(x, t)` function.
        eta: the hyperparameter for DDIM, a value of `eta=0.0` is deterministic 'probability ODE' solver, `eta=1.0` is DDPMVE.
    """
    super().__init__(num_steps, dt, epsilon)
    self.eta = eta
    self.model = model
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = jnp.exp(
        jnp.linspace(jnp.log(self.sigma_min),
                    jnp.log(self.sigma_max),
                    self.num_steps))
    self.discrete_sigmas_prev = jnp.append(0.0, self.discrete_sigmas[:-1])

  def get_estimate_x_0_vmap(self, shape, observation_map):

    def estimate_x_0(x, t, timestep):
      std = self.discrete_sigmas[timestep]
      x = x.reshape(shape)
      x = jnp.expand_dims(x, axis=0)
      t = jnp.expand_dims(t, axis=0)
      epsilon = self.model(x, t)
      epsilon = epsilon.flatten()
      x = x.flatten()
      x_0 = x - std * epsilon
      return observation_map(x_0), (epsilon, x_0)
    return estimate_x_0

  def get_estimate_x_0(self, observation_map):

    def estimate_x_0(x, t, timestep):
      std = self.discrete_sigmas[timestep]
      epsilon = self.model(x, t)
      epsilon = epsilon.flatten()
      x = x.flatten()
      x_0 = x - batch_mul(std, epsilon)
      return observation_map(x_0), (epsilon, x_0)
    return estimate_x_0

  def prior(self, rng, shape):
    return random.normal(rng, shape) * self.sigma_max

  def posterior(self, x, t):
    timestep = (t * (self.num_steps - 1) / self.t1).astype(jnp.int32)
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
