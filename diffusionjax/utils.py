import jax.numpy as jnp
from jax.lax import scan
from jax import vmap
import jax.random as random
from functools import partial


def batch_mul(a, b):
    return vmap(lambda a, b: a * b)(a, b)


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


def get_score(sde, model, params, score_scaling):
    if score_scaling is True:
        return lambda x, t: -batch_mul(model.apply(params, x, t), 1. / sde.marginal_prob(x, t)[1])
    else:
        return lambda x, t: -model.apply(params, x, t)
