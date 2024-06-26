import pytest
from diffusionjax.utils import (
  batch_mul,
  get_times,
  get_linear_beta_function,
  get_timestep,
  continuous_to_discrete,
  get_exponential_sigma_function,
  get_karras_sigma_function,
  get_karras_gamma_function,
)
import jax.numpy as jnp
from jax import vmap


def test_batch_mul():
  """Placeholder test for `:meth:batch_mul` to test CI"""
  a = jnp.ones((2,)) * 2.0
  bs = [jnp.zeros((2,)), jnp.ones((2,)), jnp.ones((2,)) * jnp.pi]
  c_expecteds = [jnp.zeros((2,)), 2.0 * jnp.ones((2,)), 2.0 * jnp.ones((2,)) * jnp.pi]
  for i, b in enumerate(bs):
    c = batch_mul(a, b)
    assert jnp.allclose(c, c_expecteds[i])


def test_continuous_discrete_equivalence_linear_beta_schedule():
  beta_min = 0.1
  beta_max = 20.0
  num_steps = 1000
  # https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sde_lib.py#L127
  # expected_discrete_betas = jnp.linspace(beta_min / num_steps, beta_max / num_steps, num_steps)  # I think this is incorrect unless training in discrete time
  ts, dt = get_times(num_steps)
  beta, _ = get_linear_beta_function(beta_min=0.1, beta_max=20.0)
  actual_discrete_betas = continuous_to_discrete(vmap(beta)(ts), dt)
  expected_discrete_betas = dt * (beta_min + ts * (beta_max - beta_min))
  assert jnp.allclose(expected_discrete_betas, actual_discrete_betas)


def test_exponential_sigma_schedule():
  num_steps = 1000
  sigma_min = 0.01
  sigma_max = 378.0
  ts, dt = get_times(num_steps)
  sigma = get_exponential_sigma_function(sigma_min=sigma_min, sigma_max=sigma_max)
  actual_discrete_sigmas = vmap(sigma)(ts)
  # https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sde_lib.py#L222
  # expected_sigmas = jnp.exp(  # I think this is wrong
  #     jnp.linspace(jnp.log(sigma_min),
  #                  jnp.log(sigma_max),
  #                  num_steps))
  #
  ts, _ = get_times(num_steps, dt)
  expected_discrete_sigmas = jnp.exp(
    jnp.log(sigma_min) + ts * (jnp.log(sigma_max) - jnp.log(sigma_min))
  )

  assert jnp.allclose(expected_discrete_sigmas, actual_discrete_sigmas)


def test_karras_sigma_schedule():
  num_steps = 1000
  sigma_min = 0.002
  sigma_max = 80
  rho = 7

  # NOTE The default ts to use is `diffusionjax.utils.get_times(num_steps, t0=0.0)`.
  ts, _ = get_times(num_steps, t0=0.0)
  ts = ts.flatten()
  sigma = get_karras_sigma_function(
    sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
  actual_discrete_sigmas = vmap(sigma)(ts)

  step_indices = jnp.arange(num_steps)
  expected_discrete_sigmas = (
    sigma_max ** (1 / rho)
    + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
  ) ** rho
  expected_discrete_sigmas = jnp.flip(expected_discrete_sigmas)
  assert jnp.allclose(expected_discrete_sigmas, actual_discrete_sigmas)


def test_get_timestep_continuous():
  def unit(ts):
    t0 = ts[0]
    t1 = ts[-1]
    t = ts[0]
    num_steps = jnp.size(ts)
    timestep = get_timestep(t, t0, t1, num_steps)
    assert timestep == 0

    t = ts[-1]
    timestep = get_timestep(t, t0, t1, num_steps)
    assert timestep == num_steps - 1

    t = ts[num_steps - num_steps // 2]
    timestep = get_timestep(t, t0, t1, num_steps)
    assert timestep == num_steps - num_steps // 2

  ts, dt = get_times()
  ts = ts.flatten()
  assert jnp.size(ts) == 1000
  assert jnp.isclose(ts[1] - ts[0], 0.001)
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.001
  assert ts[-1] == 1.0
  unit(ts)

  ts, dt = get_times(dt=0.1)
  ts = ts.flatten()
  assert jnp.size(ts) == 1000
  assert jnp.isclose(ts[1] - ts[0], 0.1)
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.1
  assert ts[-1] == 0.1 * 1000
  unit(ts)

  ts, dt = get_times(t0=0.01)
  ts = ts.flatten()
  assert jnp.size(ts) == 1000
  assert jnp.isclose(ts[1] - ts[0], (1.0 - 0.01) / (1000 - 1))
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.01
  assert ts[-1] == 1.0
  unit(ts)

  ts, dt = get_times(dt=0.1, t0=0.01)
  ts = ts.flatten()
  assert jnp.size(ts) == 1000
  assert jnp.isclose(ts[1] - ts[0], 0.1)
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.01
  assert ts[-1] == 0.1 * (1000 - 1) + 0.01
  unit(ts)

  ts, dt = get_times(num_steps=100, dt=0.1, t0=0.01)
  ts = ts.flatten()
  assert jnp.size(ts) == 100
  assert jnp.isclose(ts[1] - ts[0], 0.1)
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.01
  assert ts[-1] == 0.1 * (100 - 1) + 0.01
  unit(ts)

  # Catch any rounding errors for low number of steps

  ts, dt = get_times(num_steps=10)
  ts = ts.flatten()
  assert jnp.size(ts) == 10
  assert ts[1] - ts[0] == 0.1
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.1
  assert ts[-1] == 1.0
  unit(ts)

  ts, dt = get_times(dt=0.05, num_steps=10)
  ts = ts.flatten()
  assert jnp.size(ts) == 10
  assert ts[1] - ts[0] == 0.05
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.05
  assert ts[-1] == 0.05 * 10
  unit(ts)

  ts, dt = get_times(t0=0.01, num_steps=10)
  ts = ts.flatten()
  assert jnp.size(ts) == 10
  assert jnp.isclose(ts[1] - ts[0], (1.0 - 0.01) / (10 - 1))
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.01
  assert ts[-1] == 1.0
  unit(ts)

  ts, dt = get_times(dt=0.1, t0=0.01, num_steps=10)
  ts = ts.flatten()
  assert jnp.size(ts) == 10
  assert ts[1] - ts[0] == 0.1
  assert jnp.isclose(ts[1] - ts[0], dt)
  assert ts[0] == 0.01
  assert ts[-1] == 0.1 * (10 - 1) + 0.01
  unit(ts)


def test_karras_gamma_function(s_churn=50.0, s_min=10.0, s_max=float('inf')):
  num_steps = 100
  ts, _ = get_times(num_steps, t0=0.0)
  ts = ts.flatten()
  sigma = get_karras_sigma_function(sigma_min=0.01, sigma_max=80.0, rho=7)
  sigmas = vmap(sigma)(ts)
  gamma = get_karras_gamma_function(num_steps, s_churn=s_churn, s_min=s_min, s_max=s_max)
  gammas = gamma(sigmas)
  for gamma_actual, sigma_actual in zip(gammas, sigmas):
    gamma_expected = min(s_churn / num_steps, jnp.sqrt(2) - 1) if s_min <= sigma_actual <= s_max else 0.0
    assert gamma_actual == gamma_expected


test_karras_gamma_function()

