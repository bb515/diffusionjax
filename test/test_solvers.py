import pytest
from diffusionjax.solvers import EulerMaruyama
from diffusionjax.sde import VP
# from diffusionjax.utils import batch_mul
import jax.numpy as jnp


def test_get_timestep_continuous():
  sde = VP()

  def get_timestep(t, solver):
    # Test to make sure this fundamental conversion between continuous time t and discrete timestep always works
    return ((t - solver.epsilon) * (solver.num_steps - 1) / (solver.t1 - solver.epsilon)).astype(jnp.int32)[0]

  def unit(solver):
    t = solver.ts[0]
    timestep = get_timestep(t, solver)
    assert timestep == 0

    t = solver.ts[-1]
    timestep = get_timestep(t, solver)
    assert timestep == solver.num_steps - 1

    t = solver.ts[solver.num_steps - solver.num_steps//2]
    timestep = get_timestep(t, solver)
    assert timestep == solver.num_steps - solver.num_steps//2

  solver = EulerMaruyama(sde)
  assert solver.num_steps == 1000
  assert solver.dt == 0.001
  assert solver.epsilon == 0.001
  assert solver.t1 == 1.0
  unit(solver)

  solver = EulerMaruyama(sde, dt=0.1)
  assert solver.num_steps == 1000
  assert solver.dt == 0.1
  assert solver.epsilon == 0.1
  assert solver.t1 == 0.1 * 1000
  unit(solver)

  solver = EulerMaruyama(sde, epsilon=0.01)
  assert solver.num_steps == 1000
  assert jnp.isclose(solver.dt, (1.0 - 0.01) / (1000 - 1))
  assert solver.epsilon == 0.01
  assert solver.t1 == 1.0
  unit(solver)

  solver = EulerMaruyama(sde, dt=0.1, epsilon=0.01)
  assert solver.num_steps == 1000
  assert solver.dt == 0.1
  assert solver.epsilon == 0.01
  assert solver.t1 == 0.1 * (1000 - 1) + 0.01
  unit(solver)

  solver = EulerMaruyama(sde, num_steps=100, dt=0.1, epsilon=0.01)
  assert solver.num_steps == 100
  assert jnp.isclose(solver.dt, 0.1)
  assert solver.epsilon == 0.01
  assert solver.t1 == 0.1 * (100 - 1) + 0.01
  unit(solver)

  # Catch any rounding errors for low number of steps
  solver = EulerMaruyama(sde, num_steps=10)
  assert solver.num_steps == 10
  assert solver.dt == 0.1
  assert solver.epsilon == 0.1
  assert solver.t1 == 1.0
  unit(solver)

  solver = EulerMaruyama(sde, dt=0.05, num_steps=10)
  assert solver.num_steps == 10
  assert solver.dt == 0.05
  assert solver.epsilon == 0.05
  assert solver.t1 == 0.05 * 10
  unit(solver)

  solver = EulerMaruyama(sde, epsilon=0.01, num_steps=10)
  assert solver.num_steps == 10
  assert jnp.isclose(solver.dt, (1.0 - 0.01) / (10 - 1))
  assert solver.epsilon == 0.01
  assert solver.t1 == 1.0
  unit(solver)

  solver = EulerMaruyama(sde, dt=0.1, epsilon=0.01, num_steps=10)
  assert solver.num_steps == 10
  assert solver.dt == 0.1
  assert solver.epsilon == 0.01
  assert solver.t1 == 0.1 * (10 - 1) + 0.01
  unit(solver)
