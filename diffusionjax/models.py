import flax.linen as nn
import jax.numpy as jnp
from math import prod


class MLP(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
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

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t


class CNN(nn.Module):
    """
    A simple model with a single convolutional layer with time as another channel.
    Functions are designed for a mini-batch of inputs.
    """
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        ndim = x.ndim
        t = t.reshape((t.shape[0],) + (1,) * (ndim - 1))
        t = jnp.tile(t, (1,) + x_shape[1:-1] + (1,))
        # Add time as another channel
        x = jnp.concatenate((x, t), axis=-1)
        # Global convolution
        x = nn.Conv(x_shape[-1], kernel_size=(9,) * (ndim - 2))(x)
        return x

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t
