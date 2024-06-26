import pytest
from diffusionjax.utils import (
  batch_mul,
  get_times,
  get_linear_beta_function,
  get_timestep,
  continuous_to_discrete,
  get_exponential_sigma_function,
  get_karras_sigma_function,
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

  # TODO: The default ts to use is `diffusionjax.utils.get_times(num_steps, t0=0.0)`.
  ts, dt = get_times(num_steps)

  sigma = get_karras_sigma_function(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
  ts = jnp.concatenate([jnp.array([[0.0]]), ts])

  actual_discrete_sigmas = vmap(sigma)(ts)
  # https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sde_lib.py#L222
  # expected_sigmas = jnp.exp(  # I think this is wrong
  #     jnp.linspace(jnp.log(sigma_min),
  #                  jnp.log(sigma_max),
  #                  num_steps))
  #
  step_indices = jnp.arange(num_steps)
  ts, _ = get_times(num_steps, dt)

  # Time step discretization.
  print(step_indices)
  expected_discrete_sigmas = (
    sigma_max ** (1 / rho)
    + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
  ) ** rho
  print(expected_discrete_sigmas.shape)
  print(expected_discrete_sigmas[:1].shape)

  expected_discrete_sigmas = jnp.concatenate(
    [expected_discrete_sigmas, jnp.zeros_like(expected_discrete_sigmas[:1])]
  )  # sigma_N = 0

  print(expected_discrete_sigmas.shape)
  expected_discrete_sigmas = jnp.flip(expected_discrete_sigmas)

  import matplotlib.pyplot as plt

  plt.plot(expected_discrete_sigmas)
  plt.savefig("expected.png")
  plt.close()

  plt.plot(actual_discrete_sigmas)
  plt.savefig("actual.png")
  plt.close()

  ts = jnp.concatenate([jnp.array([[0.0]]), ts]).flatten()
  actual_discrete_sigmas = actual_discrete_sigmas.flatten()

  print(ts.shape)
  print(expected_discrete_sigmas.shape)
  print(actual_discrete_sigmas.shape)

  plt.plot(
    ts,
    (expected_discrete_sigmas - actual_discrete_sigmas) / expected_discrete_sigmas,
  )
  plt.savefig("abs.png")
  plt.close()

  plt.plot(ts, actual_discrete_sigmas / expected_discrete_sigmas)
  plt.savefig("rel.png")
  plt.close()

  assert jnp.allclose(expected_discrete_sigmas, actual_discrete_sigmas)
  assert 0

  import jax.random as random

  rng = random.PRNGKey(2023)
  rng, step_rng = random.split(rng, 2)

  S_churn = 0
  S_min = 0
  S_max = float("inf")
  S_noise = 1

  def denoise(x_hat, t_hat):
    return x_hat

  noise = 0.0
  x_next = noise * expected_discrete_sigmas[0]
  print(zip(expected_discrete_sigmas[:-1], expected_discrete_sigmas[1:]))
  assert 0


def test_karras_heun_sampler():
  # TODO

  def sample_heun(denoise, discrete_sigmas, num_steps):
    for i, (t_cur, t_next) in enumerate(
      zip(discrete_sigmas[:-1], discrete_sigmas[1:])
    ):  # 0, ..., N-1
      print(t_cur, t_next)
      x_cur = x_next

      # Increase noise temporarily.
      t_hat = t_cur
      x_hat = x_cur
      # note that this is not the same x_next as the one used to calculate d_prime
      # so there is actually no need to carry the score vector across

      # surely that this is not efficient? if S_churn is zero, then just carry
      # over the evaluation from the previous timestep. Is this how it worked in
      # my Heun solver, so I will want to make a new sampler for second order stuff

      # Euler step.
      d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
      # I should have precomputed the delta ts
      x_next = x_hat + (t_next - t_hat) * d_cur

      # Apply 2nd order correction. does not apply on the final step
      # since t_next would be zero, so t_next is only needed
      # for the 2nd order correction and the score is never evaluated (or trained) here
      if i < num_steps - 1:
        d_prime = (x_next - denoise(x_next, t_next)) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


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


test_continuous_discrete_equivalence_karras_sigma_schedule()
