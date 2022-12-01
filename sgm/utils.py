import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, jit, grad, value_and_grad
import jax.random as random
from functools import partial
import optax
import flax.linen as nn


#Initialize the optimizer
optimizer = optax.adam(1e-3)


class NonLinear(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        in_size = x.shape[1]
        n_hidden = 256
        act = nn.relu
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=1)
        x = jnp.concatenate([x, t], axis=1)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)  # score_t * std_t


def retrain_nn(
        update_step, num_epochs, step_rng, samples, score_model, params,
        opt_state, loss_fn, batch_size=5, decomposition=False):
    if decomposition:
        L = 2
    else:
        L = 1
    train_size = samples.shape[0]
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    mean_losses = jnp.zeros((num_epochs, L))
    for i in range(num_epochs):
        rng, step_rng = random.split(step_rng)
        perms = random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = jnp.zeros((jnp.shape(perms)[0], L))
        for j, perm in enumerate(perms):
            batch = samples[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn, has_aux=decomposition)
            if decomposition:
                loss = loss[1]
            losses = losses.at[j].set(loss)
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 1000 == 0:
            if L==1: print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
            if L==2: print("Tangent loss {:.2f}, perpendicular loss {:.2f}".format(mean_loss[0], mean_loss[1]))
    return score_model, params, opt_state, mean_losses


def get_score_fn(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: -model.evaluate(params, x, t) / sde.marginal_prob(x, t)[1]
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

