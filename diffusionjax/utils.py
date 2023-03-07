import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, jit, grad, value_and_grad
import jax.random as random
from functools import partial
import optax
from math import prod


#Initialize the optimizer
optimizer = optax.adam(1e-3)


def batch_mul(a, b):
    return vmap(lambda a, b: a * b)(a, b)


def retrain_nn(
        update_step, num_epochs, step_rng, samples, score_model, params,
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
            loss_eval, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss)
            losses = losses.at[j].set(loss_eval)
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
    return score_model, params, opt_state, mean_losses


def get_score(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: -batch_mul(model.evaluate(params, x, t), 1. / sde.marginal_prob(x, t)[1])
    else:
        return lambda x, t: -model.evaluate(params, x, t)


@partial(jit, static_argnums=[4, 5, 6])
def update_step(params, rng, batch, opt_state, model, loss, has_aux=False):
    """
    Takes the gradient of the loss function and updates the model weights (params) using it.
    Args:
        params: the current weights of the model
        rng: random number generator from jax
        batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
        opt_state: the internal state of the optimizer
        model: the score function
        loss: A loss function that can be used for score matching training.
        has_aux:
    Returns:
        The value of the loss function (for metrics), the new params and the new optimizer states function (for metrics), the new params and the new optimizer state.
    """
    val, grads = value_and_grad(loss, has_aux=has_aux)(params, model, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state

