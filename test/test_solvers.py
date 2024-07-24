import pytest
from diffusionjax.utils import (
  get_times,
  get_karras_sigma_function,
  get_karras_gamma_function,
)
import jax.numpy as jnp
import jax.random as random
from diffusionjax.solvers import EDMHeun
from diffusionjax.utils import get_sampler


def test_karras_heun_sampler():
  num_steps = 100
  sigma_min = 0.002
  sigma_max = 80
  rho = 7

  batch_size = 4
  image_size = 1
  sample_shape = (batch_size, image_size)

  s_churn = 100
  s_min = 10.0
  s_max = 60.0
  s_noise = 1

  # NOTE The default ts to use is `diffusionjax.utils.get_times(num_steps, t0=0.0)`.
  ts, _ = get_times(num_steps, t0=0.0)
  sigma = get_karras_sigma_function(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
  gamma = get_karras_gamma_function(num_steps, s_churn=s_churn, s_min=s_min, s_max=s_max)

  step_indices = jnp.arange(num_steps)
  t_steps = (
    sigma_max ** (1 / rho)
    + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
  ) ** rho

  # t_N = 0
  t_steps = jnp.append(t_steps, jnp.zeros_like(t_steps[:1]))

  def denoise(x_hat, t_hat):
    return x_hat

  def edm_sampler(rng, denoise, t_steps, num_steps):
    # Main sampling loop.
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, sample_shape)
    x_next = noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
      rng, step_rng = random.split(rng)
      x_cur = x_next

      # Increase noise temporarily.
      if s_churn > 0 and s_min <= t_cur <= s_max:
        gamma = min(s_churn / num_steps, jnp.sqrt(2) - 1)
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * s_noise * random.normal(step_rng, x_cur.shape)
      else:
        t_hat = t_cur
        x_hat = x_cur

      dt = t_next - t_hat

      # Euler step.
      d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
      x_next = x_hat + dt * d_cur

      # Apply 2nd order correction.
      if i < num_steps - 1:
        d_prime = (x_next - denoise(x_next, t_next)) / t_next
        # note that this is not necessarily the same x_next as the one used to calculate d_cur at the next step
        # so there is actually no need to carry the score vector across
        x_next = x_hat + dt * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

  solver = EDMHeun(
    denoise=denoise, sigma=sigma, gamma=gamma, ts=ts, s_noise=s_noise)

  sampler = get_sampler(
    (batch_size, image_size),
    solver,
    stack_samples=False,
  )

  rng0 = random.PRNGKey(2023)

  x_expected = edm_sampler(rng0, denoise, t_steps, num_steps)
  x_actual, no_function_evaluations = sampler(rng0)
  # TODO: number_function_evaluations is incorrect and needs to be multiplied by two due to Heun step
  assert jnp.allclose(x_expected, x_actual)


