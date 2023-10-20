"""Benchmark of example.py training time, sample time and regression test of summary statistic of samples."""
import pytest
import jax
from jax import jit, vmap, grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from diffusionjax.run_lib import get_model, train, get_solver
from diffusionjax.utils import get_score, get_sampler
import diffusionjax.sde as sde_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
from flax import serialization
import time
import os

# Dependencies:
# This test requires optax, https://optax.readthedocs.io/en/latest/
# This test requires orbax, https://orbax.readthedocs.io/en/latest/
# This test requires torch[cpu], https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", './configs/example.py', "Training configuration.",
  lock_config=True)
flags.mark_flags_as_required(["config"])


class CircleDataset(Dataset):
  """Dataset containing samples from the circle."""
  def __init__(self, num_samples):
    self.train_data = self.sample_circle(num_samples)

  def __len__(self):
    return self.train_data.shape[0]

  def __getitem__(self, idx):
    return self.train_data[idx]

  def sample_circle(self, num_samples):
    """Samples from the unit circle, angles split.

    Args:
      num_samples: The number of samples.

    Returns:
      An (num_samples, 2) array of samples.
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

  def metric_names(self):
    return ['mean']

  def calculate_metrics_batch(self, batch):
    return vmap(lambda x: jnp.mean(x, axis=0))(batch)[0, 0]

  def get_data_scaler(self, config):
    def data_scaler(x):
      return x / jnp.sqrt(2)
    return data_scaler

  def get_data_inverse_scaler(self, config):
    def data_inverse_scaler(x):
      return x * jnp.sqrt(2)
    return data_inverse_scaler


def main(argv):
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.training.pmap else 1
  rng = random.PRNGKey(config.seed)

  # Setup SDE
  if config.training.sde.lower()=='vpsde':
    sde = sde_lib.VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
  elif config.training.sde.lower()=='vesde':
    sde = sde_lib.VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
  else:
    raise NotImplementedError(f"SDE {config.training.SDE} unknown.")

  # Build data iterators
  num_samples = 8
  dataset = CircleDataset(num_samples=num_samples)
  scaler = dataset.get_data_scaler(config)
  inverse_scaler = dataset.get_data_inverse_scaler(config)

  time_prev = time.time()
  params, _, mean_losses = train(
    (config.training.batch_size//jax.local_device_count(), config.data.image_size),
    config, dataset, workdir=None, use_wandb=False)
  train_time_delta = time.time() - time_prev
  expected_train_time = 7.0  # seconds
  print("train time: {}s".format(train_time_delta))
  expected_mean_loss = 0.4081565
  mean_loss = jnp.mean(mean_losses)
  import matplotlib.pyplot as plt
  plt.plot(mean_losses)
  plt.show()

  # Get trained score
  trained_score = get_score(sde, get_model(config), params, score_scaling=config.training.score_scaling)
  outer_solver, inner_solver = get_solver(config, sde, trained_score)
  sampler = get_sampler(
    (config.eval.batch_size // num_devices, config.data.image_size),
    outer_solver,
    inner_solver,
    denoise=config.sampling.denoise,
    inverse_scaler=inverse_scaler)

  if config.training.pmap:
    sampler = jax.pmap(sampler, axis_name='batch')
    rng, *sample_rng = random.split(rng, 1 + num_devices)
    sample_rng = jnp.asarray(sample_rng)
  else:
    rng, sample_rng = random.split(rng, 2)

  time_prev = time.time()
  q_samples, _ = sampler(sample_rng)
  sample_time_delta = time.time() - time_prev
  print("sample time: {}s".format(sample_time_delta))
  expected_sample_time = 1.25 # seconds
  # TODO: Is there a machine agnostic way to do unit tests on benchmark scores?
  q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size)
  plt.scatter(q_samples[:, 0], q_samples[:, 1])
  plt.show()
  radii = jnp.linalg.norm(q_samples, axis=1)
  expected_mean_radii = 1.0236381
  mean_radii = jnp.mean(radii)
  expected_std_radii = 0.09904917
  std_radii = jnp.std(radii)

  # Benchmark
  assert jnp.isclose(train_time_delta, expected_train_time, rtol=0.035), "train (got {}s, expected {}s)".format(train_time_delta, expected_train_time)
  assert jnp.isclose(sample_time_delta, expected_sample_time, rtol=0.12), "sample (got {}s, expected {}s)".format(sample_time_delta, expected_sample_time)

  # Regression
  assert jnp.isclose(mean_radii, expected_mean_radii)
  assert jnp.isclose(std_radii, expected_std_radii)
  assert jnp.isclose(mean_loss, expected_mean_loss), "average loss (got {}, expected {})".format(mean_loss, expected_mean_loss)


if __name__ == "__main__":
  app.run(main)
