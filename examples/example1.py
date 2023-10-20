"""Diffusion models introduction. An example using 1 dimensional image data."""
from jax import vmap, jit, grad, value_and_grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import serialization
from functools import partial
import matplotlib.pyplot as plt
from diffusionjax.plot import plot_score, plot_heatmap
from diffusionjax.utils import get_score, get_loss, get_sampler
from diffusionjax.inverse_problems import get_inpainter, get_projection_sampler
from diffusionjax.solvers import EulerMaruyama
from diffusionjax.models import MLP
from diffusionjax.sde import VE
import numpy as np
import os

# Dependencies:
# This example requires mlkernels package, https://github.com/wesselb/mlkernels#installation
import lab as B
from mlkernels import Matern52
# This example requires optax, https://optax.readthedocs.io/en/latest/
import optax


x_max = 5.0
epsilon = 1e-4


#Initialize the optimizer
optimizer = optax.adam(1e-3)


@partial(jit, static_argnums=[4])
def update_step(params, rng, batch, opt_state, loss):
  """
  Takes the gradient of the loss function and updates the model weights (params) using it.
  Args:
      params: the current weights of the model
      rng: random number generator from jax
      batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
      opt_state: the internal state of the optimizer
      loss: A loss function that can be used for score matching training.
  Returns:
      The value of the loss function (for metrics), the new params and the new optimizer states function (for metrics),
      the new params and the new optimizer state.
  """
  val, grads = value_and_grad(loss)(params, rng, batch)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return val, params, opt_state


def retrain_nn(
    update_step, num_epochs, step_rng, samples, params,
    opt_state, loss, batch_size=5):
  train_size = samples.shape[0]
  batch_size = min(train_size, batch_size)
  steps_per_epoch = train_size // batch_size
  mean_losses = jnp.zeros((num_epochs, 1))
  for i in range(num_epochs):
    rng, step_rng = random.split(step_rng)
    perms = random.permutation(step_rng, train_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    losses = jnp.zeros((jnp.shape(perms)[0], 1))
    for j, perm in enumerate(perms):
      batch = samples[perm, :]
      rng, step_rng = random.split(rng)
      loss_eval, params, opt_state = update_step(params, step_rng, batch, opt_state, loss)
      losses = losses.at[j].set(loss_eval)
    mean_loss = jnp.mean(losses, axis=0)
    mean_losses = mean_losses.at[i].set(mean_loss)
    if i % 10 == 0:
      print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
  return params, opt_state, mean_losses


def sample_image_rgb(rng, num_samples, image_size, kernel, num_channels=1):
  """Samples from a GMRF"""
  x = np.linspace(-x_max, x_max, image_size)
  x = x.reshape(image_size, 1)
  C = B.dense(kernel(x)) + epsilon * B.eye(image_size)
  u = random.multivariate_normal(rng, mean=jnp.zeros(x.shape[0]), cov=C, shape=(num_samples, num_channels))
  u = u.transpose((0, 2, 1))
  return u, C


def plot_score_ax_sample(ax, sample, score, t, area_min=-1, area_max=1, fname="plot_score"):
  @partial(jit, static_argnums=[0,])
  def helper(score, sample, t, area_min, area_max):
    x = jnp.linspace(area_min, area_max, 16)
    x, y = jnp.meshgrid(x, x)
    grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
    sample = jnp.tile(sample, (len(x.flatten()), 1, 1, 1))
    sample.at[:, [0, 1], 0, 0].set(grid)
    t = jnp.ones((grid.shape[0],)) * t
    scores = score(sample, t)
    return grid, scores
  grid, scores = helper(score, sample, t, area_min, area_max)
  ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0, 0, 0], scores[:, 1, 0, 0])


def plot_samples_1D(samples, image_size, fname="samples 1D.png"):
  x = np.linspace(-x_max, x_max, image_size)
  plt.plot(x, samples[:, :, 0].T)
  plt.savefig(fname)
  plt.close()


def main():
  num_epochs = 128
  rng = random.PRNGKey(2023)
  rng, step_rng = random.split(rng, 2)
  num_samples = 18000
  num_channels = 1
  image_size = 64  # image size

  samples, C = sample_image_rgb(rng, num_samples=num_samples, image_size=image_size, kernel=Matern52(), num_channels=num_channels)  # (num_samples, image_size, num_channels)

  # Reshape image data
  samples = samples.reshape(-1, image_size, num_channels)
  plot_samples_1D(samples[:64], image_size, "samples")

  # Get sde model
  sde = VE(sigma_min=0.01, sigma_max=3.0)

  def nabla_log_pt(x, t):
    """Score.

    Returns:
      The true log density.
      .. math::
        \nabla_{x} \log p_{t}(x)
    """
    x_shape = x.shape
    v_t = sde.variance(t)
    m_t = sde.mean_coeff(t)
    x = x.flatten()
    score = - jnp.linalg.solve(m_t**2 * C + v_t * jnp.eye(x_shape[0]), x)
    return score.reshape(x_shape)

  # Neural network training via score matching
  batch_size = 64
  score_model = MLP()

  # Initialize parameters
  params = score_model.init(step_rng, jnp.zeros((batch_size, image_size, num_channels)), jnp.ones((batch_size,)))

  # Initialize optimizer
  opt_state = optimizer.init(params)

  if not os.path.exists('/tmp/output1'):
    solver = EulerMaruyama(sde)

    # Get loss function
    loss = get_loss(
      sde, solver, score_model, score_scaling=True, likelihood_weighting=False,
      reduce_mean=True, pointwise_t=False)

    # Train with score matching
    params, opt_state, _ = retrain_nn(
      update_step=update_step,
      num_epochs=num_epochs,
      step_rng=step_rng,
      samples=samples,
      params=params,
      opt_state=opt_state,
      loss=loss,
      batch_size=batch_size)

    # Save params
    output = serialization.to_bytes(params)
    f = open('/tmp/output1', 'wb')
    f.write(output)
  else:  # Load pre-trained model parameters
    f = open('/tmp/output1', 'rb')
    output = f.read()
    params = serialization.from_bytes(params, output)

  # Get trained score
  trained_score = get_score(sde, score_model, params, score_scaling=True)
  solver = EulerMaruyama(sde.reverse(trained_score))
  sampler = get_sampler((512, image_size, num_channels), solver, denoise=True)

  rng, sample_rng = random.split(rng, 2)
  q_samples, num_function_evaluations = sampler(sample_rng)

  # C_emp = jnp.corrcoef(q_samples[:, :, 0].T)
  # delta = jnp.linalg.norm(C - C_emp) / image_size

  plot_samples_1D(q_samples[:64], image_size=image_size, fname="samples trained score")
  plot_heatmap(samples=q_samples[:, [0, 1], 0], area_bounds=[-3., 3.], fname="heatmap trained score")

  # Condition on one of the coordinates
  data = jnp.zeros((image_size, num_channels))
  data = data.at[[0, -1], 0].set([-1., 1.])
  mask = jnp.zeros((image_size, num_channels), dtype=float)
  mask = mask.at[[0, -1], 0].set([1., 1.])
  data = jnp.tile(data, (5, 1, 1))
  mask = jnp.tile(mask, (5, 1, 1))

  # Get inpainter
  inpainter = get_inpainter(solver, stack_samples=False)
  rng, sample_rng = random.split(rng, 2)
  q_samples, _ = inpainter(sample_rng, data, mask)
  plot_samples_1D(q_samples, image_size=image_size, fname="samples inpainted")

  # Get projection sampler
  projection_sampler = get_projection_sampler(solver, stack_samples=False)
  rng, sample_rng = random.split(rng, 2)
  q_samples, _ = projection_sampler(sample_rng, data, mask, 1e-2)
  plot_samples_1D(q_samples, image_size=image_size, fname="samples projected")


if __name__ == "__main__":
  main()
