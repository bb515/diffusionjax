"""Diffusion models introduction. An example using 2 dimensional image data."""
import jax
from jax import vmap, jit, grad, value_and_grad
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import serialization
from functools import partial
import matplotlib.pyplot as plt
from diffusionjax.plot import plot_samples, plot_heatmap
from diffusionjax.utils import get_score, get_loss, get_sampler
from diffusionjax.solvers import EulerMaruyama, Annealed
from diffusionjax.models import CNN
from diffusionjax.sde import VE, udlangevin
import numpy as np
import os

# Dependencies:
# This example requires mlkernels package, https://github.com/wesselb/mlkernels#installation
from mlkernels import Matern52
import lab as B
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


def image_grid(x, image_size, num_channels):
  img = x.reshape(-1, image_size, image_size, num_channels)
  w = int(np.sqrt(img.shape[0]))
  return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
  img = image_grid(x, image_size, num_channels)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.savefig(fname)
  plt.close()


def sample_image_rgb(rng, num_samples, image_size, kernel, num_channels):
  """Samples from a GMRF."""
  x = np.linspace(-x_max, x_max, image_size)
  y = np.linspace(-x_max, x_max, image_size)
  xx, yy = np.meshgrid(x, y)
  xx = xx.reshape(image_size**2, 1)
  yy = yy.reshape(image_size**2, 1)
  z = np.hstack((xx, yy))
  C = B.dense(kernel(z))  + epsilon * B.eye(image_size**2)
  u = random.multivariate_normal(rng, mean=jnp.zeros(xx.shape[0]), cov=C, shape=(num_samples, num_channels))
  u = u.transpose((0, 2, 1))
  return u, C


def plot_samples_1D(samples, image_size, fname="samples 1D.png"):
  x = np.linspace(-x_max, x_max, image_size)
  plt.plot(x, samples[:, :, 0, 0].T)
  plt.savefig(fname)
  plt.close()


def main():
  num_epochs = 200
  rng = random.PRNGKey(2023)
  rng, step_rng = random.split(rng, 2)
  num_samples = 144
  num_channels = 1
  image_size = 32  # image size
  num_steps = 1000

  # Get and handle image data
  samples, _ = sample_image_rgb(rng, num_samples=num_samples, image_size=image_size, kernel=Matern52(), num_channels=num_channels)  # (num_samples, image_size**2, num_channels)
  plot_samples(samples[:64], image_size=image_size, num_channels=num_channels)
  samples = samples.reshape(-1, image_size, image_size, num_channels)
  plot_samples_1D(samples[:64], image_size, "samples 1D")

  # Get sde model
  sde = VE(sigma_min=0.001, sigma_max=3.0)

  # Neural network training via score matching
  batch_size = 16
  score_model = CNN()

  # Initialize parameters
  params = score_model.init(step_rng, jnp.zeros((batch_size, image_size, image_size, num_channels)), jnp.ones((batch_size,)))

  # Initialize optimizer
  opt_state = optimizer.init(params)

  if not os.path.exists('/tmp/output2'):
    # Get loss function
    solver = EulerMaruyama(sde, num_steps=num_steps)
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
    f = open('/tmp/output2', 'wb')
    f.write(output)
  else:  # Load pre-trained model parameters
    f = open('/tmp/output2', 'rb')
    output = f.read()
    params = serialization.from_bytes(params, output)

  # Get trained score
  trained_score = get_score(sde, score_model, params, score_scaling=True)

  # Get the outer loop of a numerical solver, also known as "predictor"
  rsde = sde.reverse(trained_score)
  outer_solver = EulerMaruyama(rsde, num_steps=num_steps)

  # Get the inner loop of a numerical solver, also known as "corrector"
  inner_solver = Annealed(rsde.correct(udlangevin), num_steps=2, snr=0.01)

  # pmap across devices. pmap assumes devices are identical model. If this is not the case,
  # use the devices argument in pmap
  num_devices =  jax.local_device_count()
  sampler = jax.pmap(
    get_sampler((64//num_devices, image_size, image_size, num_channels), outer_solver, inner_solver, denoise=True),
    axis_name='batch',
    # devices = jax.devices()[:],
  )
  rng, *sample_rng = random.split(rng, num_devices + 1)
  sample_rng = jnp.asarray(sample_rng)
  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(64, image_size, image_size, num_channels)
  plot_samples(q_samples, image_size=image_size, num_channels=num_channels, fname="samples trained score")
  plot_samples_1D(q_samples, image_size, fname="samples 1D trained score")
  plot_heatmap(samples=q_samples[:, [0, 1], 0, 0], area_bounds=[-3., 3.], fname="heatmap trained score")


if __name__ == "__main__":
  main()
