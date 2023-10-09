"""Functions are designed for a mini-batch of inputs."""
import flax.linen as nn
import numpy as np
import jax.numpy as jnp


class MLP(nn.Module):
  @nn.compact
  def __call__(self, x, t):
    x_shape = x.shape
    in_size = np.prod(x_shape[1:])
    n_hidden = 256
    t = t.reshape((t.shape[0], -1))
    x = x.reshape((x.shape[0], -1))  # flatten
    t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
    x = jnp.concatenate([x, t], axis=-1)
    x = nn.Dense(n_hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(n_hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(n_hidden)(x)
    x = nn.relu(x)
    x = nn.Dense(in_size)(x)
    return x.reshape(x_shape)


class CNN(nn.Module):
  @nn.compact
  def __call__(self, x, t):
    x_shape = x.shape
    ndim = x.ndim

    n_hidden = x_shape[1]
    n_time_channels = 1

    t = t.reshape((t.shape[0], -1))
    t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
    t = nn.Dense(n_hidden**2 * n_time_channels)(t)
    t = nn.relu(t)
    t = nn.Dense(n_hidden**2 * n_time_channels)(t)
    t = nn.relu(t)
    t = t.reshape(t.shape[0], n_hidden, n_hidden, n_time_channels)
    # Add time as another channel
    x = jnp.concatenate((x, t), axis=-1)
    # A single convolution layer
    x = nn.Conv(x_shape[-1], kernel_size=(9,) * (ndim - 2))(x)
    return x
