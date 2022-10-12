import flax.linen as nn
import functools
import jax.numpy as jnp
import numpy as np
import ml_collections


@utils.register_model(name='linear')
class Linear(nn.Module):
    """
    Linear score model.

    This is a temporary model for testing purposes.
    """
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, t):
        # config parsing
        config = self.config
        # act = get_act(config)  # TODO
        sigmas = utils.get_sigmas(config)
        batch_size = x.shape[0]
        N = jnp.shape(x)[1]
        in_size = (N + 1) * N
        n_hidden = 256
        h = jnp.array([t - 0.5, jnp.cos(2*jnp.pi*t)]).T
        h = nn.Dense(n_hidden)(h)
        h = nn.relu(h)
        h = nn.Dense(n_hidden)(h)
        h = nn.relu(h)
        h = nn.Dense(n_hidden)(h)
        h = nn.relu(h)
        h = nn.Dense(in_size)(h)
        h = jnp.reshape(h, (batch_size, N + 1, N))  # (batch_size, N + 1, N)
        h = jnp.einsum('ijk, ij -> ik', h, jnp.hstack((jnp.ones((batch_size, 1)), x)))
        if config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas
        return h  # (n_batch,)

