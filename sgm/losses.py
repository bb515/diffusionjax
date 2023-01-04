"""All functions related to loss computation and optimization."""
import jax.numpy as jnp
import jax.random as random
from sgm.utils import get_score_fn, batch_mul


def errors(ts, sde, score_fn, rng, batch, likelihood_weighting=True):
    """
    Args:
        ts: JAX tensor of time
        sde: instantiation of a valid SDE class
        score_fn: a function taking in (x, t) and returning the score
        rng: random number generator from jax
        batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    Returns:
        A random (MC) approximation to the (likelihood weighted) score errors.
    """
    mean, std = sde.marginal_prob(batch, ts)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean + batch_mul(std, noise)
    if not likelihood_weighting:
        return noise + batch_mul(score_fn(x_t, ts), std)
    else:
        return noise / std + score_fn(x_t, ts)


def get_loss_fn(sde, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for score matching training.
    Args:
        sde: instantiation of a valid SDE class.
        model: a valid flax neural network `:class:flax.linen.Module` class
        score_scaling: Boolean variable, set to `True` if learning a score scaled by the marginal standard deviation.
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
        reduce_mean: Boolean variable, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
        pointwise_t: Boolean variable, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

    Returns:
        A loss function that can be used for score matching training.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss_fn(t, params, model, rng, batch):
            n_batch = batch.shape[0]
            ts = jnp.ones((n_batch,)) * t
            score_fn = get_score_fn(sde, model, params, score_scaling)
            e = errors(ts, sde, score_fn, rng, batch, likelihood_weighting)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), ts)[1]
                e = e * g
            return reduce_op(jnp.sum(e.reshape((e.shape[0], -1))**2, axis=-1))
    else:
        def loss_fn(params, model, rng, batch):
            rng, step_rng = random.split(rng)
            n_batch = batch.shape[0]
            ts = random.randint(step_rng, (n_batch,), 1, sde.n_steps) / (sde.n_steps - 1)
            score_fn = get_score_fn(sde, model, params, score_scaling)
            e = errors(ts, sde, score_fn, rng, batch, likelihood_weighting)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), ts)[1]
                e = e * g
            return reduce_op(jnp.sum(e.reshape((e.shape[0], -1))**2, axis=-1))
    return loss_fn

