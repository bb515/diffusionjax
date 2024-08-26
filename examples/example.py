"""Diffusion models introduction.

Based off the Jupyter notebook: https://jakiw.com/sgm_intro
A tutorial on the theoretical and implementation aspects of score-based generative models, also called diffusion models.
"""

# Uncomment to enable double precision
# from jax.config import config as jax_config
# jax_config.update("jax_enable_x64", True)
import jax
from jax import jit, vmap, grad
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import logsumexp
import numpy as np
from diffusionjax.run_lib import train
from diffusionjax.utils import get_score, get_sampler, get_times
from diffusionjax.solvers import EulerMaruyama, Inpainted
from diffusionjax.inverse_problems import get_pseudo_inverse_guidance, get_vjp_guidance
from diffusionjax.plot import plot_scatter, plot_score, plot_heatmap
import diffusionjax.sde as sde_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
from flax import serialization
import time
import os

# Dependencies:
# This example requires optax, https://optax.readthedocs.io/en/latest/
# This example requires orbax, https://orbax.readthedocs.io/en/latest/
# This example requires torch[cpu], https://pytorch.org/get-started/locally/
from torch.utils.data import Dataset


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", "./configs/example.py", "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", "./examples/", "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])


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
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1 / num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

  def metric_names(self):
    return ["mean"]

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


class MLP(nn.Module):
  @nn.compact
  def __call__(self, x, t):
    x_shape = x.shape
    in_size = np.prod(x_shape[1:])
    n_hidden = 256
    t = t.reshape((t.shape[0], -1))
    x = x.reshape((x.shape[0], -1))  # flatten
    t = jnp.concatenate([t - 0.5, jnp.cos(2 * jnp.pi * t)], axis=-1)
    x = jnp.concatenate([x, t], axis=-1)
    x = nn.Dense(n_hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(n_hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(n_hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(in_size)(x)
    return x.reshape(x_shape)


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices = int(jax.local_device_count()) if config.training.pmap else 1
  rng = random.PRNGKey(config.seed)

  # Setup SDE
  if config.training.sde.lower() == "vpsde":
    from diffusionjax.utils import get_linear_beta_function

    beta, mean_coeff = get_linear_beta_function(
      beta_min=config.model.beta_min, beta_max=config.model.beta_max
    )
    sde = sde_lib.VP(beta, mean_coeff)
  elif config.training.sde.lower() == "vesde":
    from diffusionjax.utils import get_exponential_sigma_function

    sigma = get_exponential_sigma_function(
      sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max
    )
    sde = sde_lib.VE(sigma)
  else:
    raise NotImplementedError(f"SDE {config.training.SDE} unknown.")

  # Build data iterators
  num_samples = 8
  dataset = CircleDataset(num_samples=num_samples)
  scaler = dataset.get_data_scaler(config)
  inverse_scaler = dataset.get_data_inverse_scaler(config)
  plot_scatter(
    samples=dataset.train_data, index=(0, 1), fname="samples", lims=((-3, 3), (-3, 3))
  )

  def log_hat_pt(x, t):
    """Empirical distribution score.

    Args:
      x: One location in $\mathbb{R}^2$
      t: time
    Returns:
      The empirical log density, as described in the Jupyter notebook
      .. math::
        \log\hat{p}_{t}(x)
    """
    mean_coeff = sde.mean_coeff(t)  # argument t can be scalar BatchTracer or JaxArray
    mean = mean_coeff * scaler(dataset.train_data)
    std = jnp.sqrt(sde.variance(t))
    potentials = jnp.sum(-((x - mean) ** 2) / (2 * std**2), axis=1)
    return logsumexp(potentials, axis=0, b=1 / num_samples)

  # Get a jax grad function, which can be batched with vmap
  nabla_log_hat_pt = jit(vmap(grad(log_hat_pt)))

  # Running the reverse SDE with the empirical drift
  plot_score(
    score=nabla_log_hat_pt,
    scaler=scaler,
    t=0.01,
    area_bounds=[-3.0, 3],
    fname="empirical score",
  )
  ts, _ = get_times(
    num_steps=config.solver.num_outer_steps,
    dt=config.solver.dt,
    t0=config.solver.epsilon,
  )
  outer_solver = EulerMaruyama(sde.reverse(nabla_log_hat_pt), ts)
  sampler = get_sampler(
    (5760, config.data.image_size),
    outer_solver,
    denoise=config.sampling.denoise,
    stack_samples=False,
    inverse_scaler=inverse_scaler,
  )
  rng, sample_rng = random.split(rng, 2)
  q_samples, _ = sampler(sample_rng)
  plot_heatmap(
    samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap empirical score"
  )

  # What happens when I perturb the score with a constant?
  perturbed_score = lambda x, t: nabla_log_hat_pt(x, t) + 1.0
  outer_solver = EulerMaruyama(sde.reverse(perturbed_score), ts)
  sampler = get_sampler(
    (5760, config.data.image_size),
    outer_solver,
    denoise=config.sampling.denoise,
    inverse_scaler=inverse_scaler,
  )
  rng, sample_rng = random.split(rng, 2)
  q_samples, _ = sampler(sample_rng)
  plot_heatmap(
    samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap bounded perturbation"
  )

  if not os.path.exists("/tmp/output0"):
    time_prev = time.time()
    params, *_ = train(
      (config.training.batch_size // jax.local_device_count(), config.data.image_size),
      config,
      MLP(),
      dataset,
      workdir,
      use_wandb=False,
    )  # Optionally visualize results on weightsandbiases
    time_delta = time.time() - time_prev
    print("train time: {}s".format(time_delta))

    # Save params
    output = serialization.to_bytes(params)
    f = open("/tmp/output0", "wb")
    f.write(output)
  else:  # Load pre-trained model parameters
    params = MLP().init(
      rng,
      jnp.zeros(
        (config.training.batch_size // jax.local_device_count(), config.data.image_size)
      ),
      jnp.ones((config.training.batch_size // jax.local_device_count(),)),
    )
    f = open("/tmp/output0", "rb")
    output = f.read()
    params = serialization.from_bytes(params, output)

  # Get trained score
  trained_score = get_score(
    sde, MLP(), params, score_scaling=config.training.score_scaling
  )
  plot_score(
    score=trained_score,
    scaler=scaler,
    t=0.01,
    area_bounds=[-3.0, 3.0],
    fname="trained score",
  )
  outer_solver = EulerMaruyama(sde.reverse(trained_score), ts)
  sampler = get_sampler(
    (config.eval.batch_size // num_devices, config.data.image_size),
    outer_solver,
    denoise=config.sampling.denoise,
    inverse_scaler=inverse_scaler,
  )

  if config.training.pmap:
    sampler = jax.pmap(sampler, axis_name="batch")
    rng, *sample_rng = random.split(rng, 1 + num_devices)
    sample_rng = jnp.asarray(sample_rng)
  else:
    rng, sample_rng = random.split(rng, 2)

  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size)
  plot_heatmap(
    samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap trained score"
  )


  # Inverse problems
  sampling_shape = (config.eval.batch_size, config.data.image_size)
  rsde = sde.reverse(trained_score)
  # Condition on one of the coordinates
  y = jnp.array([-0.5, 0.0])
  mask = jnp.array([1.0, 0.0])
  y = scaler(y)

  # Get inpainter
  sampler = get_sampler(
    sampling_shape,
    outer_solver,
    Inpainted(rsde, mask, y),
    inverse_scaler=inverse_scaler,
    stack_samples=False,
    denoise=True,
  )
  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(sampling_shape)
  plot_heatmap(samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap inpainted")

  # Get projection sampler
  sampler = get_sampler(
    sampling_shape,
    outer_solver,
    Inpainted(rsde, mask, y),
    inverse_scaler=inverse_scaler,
    stack_samples=False,
    denoise=True,
  )
  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(sampling_shape)
  plot_heatmap(samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap projected")

  def observation_map(x):
    return mask * x

  y = jnp.tile(y, (sampling_shape[0], 1))
  # Get pseudo-inverse-guidance sampler
  sampler = get_sampler(
    sampling_shape,
    EulerMaruyama(
      sde.reverse(trained_score).guide(
        get_pseudo_inverse_guidance, observation_map, y, config.sampling.noise_std
      )
    ),
    inverse_scaler=inverse_scaler,
    stack_samples=False,
    denoise=True,
  )
  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(sampling_shape)
  plot_heatmap(samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap guided")

  y = jnp.array([-0.5, 1.0])
  H = jnp.eye(2)
  y = jnp.tile(y, (sampling_shape[0], 1))
  # Get pseudo-inverse-guidance sampler
  sampler = get_sampler(
    sampling_shape,
    EulerMaruyama(
      sde.reverse(trained_score).guide(
        get_vjp_guidance, H, y, config.sampling.noise_std, sampling_shape
      )
    ),
    inverse_scaler=inverse_scaler,
    stack_samples=False,
    denoise=True,
  )
  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(sampling_shape)
  plot_heatmap(samples=q_samples, area_bounds=[-3.0, 3.0], fname="heatmap tmpd guided")


if __name__ == "__main__":
  app.run(main)
