import jax.numpy as jnp
from jax.lax import scan, conv_general_dilated, conv_dimension_numbers
from jax import vmap, jit, grad, value_and_grad
import jax.random as random
from functools import partial
import optax
import flax.linen as nn
from math import prod


#Initialize the optimizer
optimizer = optax.adam(1e-3)


def batch_mul(a, b):
    return vmap(lambda a, b: a * b)(a, b)


class MLP(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""
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


def retrain_nn(
        update_step, num_epochs, step_rng, samples, score_model, params,
        opt_state, loss_fn, batch_size=5):
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
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn)
            losses = losses.at[j].set(loss)
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
    return score_model, params, opt_state, mean_losses


def get_score_fn(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: -batch_mul(model.evaluate(params, x, t), 1. / sde.marginal_prob(x, t)[1])
    else:
        return lambda x, t: -model.evaluate(params, x, t)


@partial(jit, static_argnums=[4, 5, 6])
def update_step(params, rng, batch, opt_state, model, loss_fn, has_aux=False):
    """
    Takes the gradient of the loss function and updates the model weights (params) using it.
    Args:
        params: the current weights of the model
        rng: random number generator from jax
        batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
        opt_state: the internal state of the optimizer
        model: the score function
        loss_fn: A loss function that can be used for score matching training.
        has_aux:
    Returns:
        The value of the loss function (for metrics), the new params and the new optimizer states function (for metrics), the new params and the new optimizer state.
    """
    val, grads = value_and_grad(loss_fn, has_aux=has_aux)(params, model, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state

