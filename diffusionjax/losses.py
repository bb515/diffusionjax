"""All functions related to loss computation and optimization."""
import jax.numpy as jnp
import jax.random as random
from diffusionjax.utils import get_score, batch_mul


def errors(ts, sde, score, rng, data, likelihood_weighting=True):
    """
    Args:
        ts: JAX array of times.
        sde: Instantiation of a valid SDE class.
        score: A function taking in (x, t) and returning the score.
        rng: Random number generator from JAX.
        data: A batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N).
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
    Returns:
        A random (MC) approximation to the (likelihood weighted) score errors.
    """
    mean, std = sde.marginal_prob(data, ts)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, data.shape)
    x_t = mean + batch_mul(std, noise)
    if not likelihood_weighting:
        return noise + batch_mul(score(x_t, ts), std)
    else:
        return batch_mul(noise, 1. / std) + score(x_t, ts)


def get_loss(sde, solver, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for score matching training.
    Args:
        sde: Instantiation of a valid SDE class.
        solver: Instantiation of a valid Solver class.
        model: A valid flax neural network `:class:flax.linen.Module` class.
        score_scaling: Boolean variable, set to `True` if learning a score scaled by the marginal standard deviation.
        likelihood_weighting: Boolean variable, set to `True` if likelihood weighting, as described in Song et al. 2020 (https://arxiv.org/abs/2011.13456), is applied.
        reduce_mean: Boolean variable, set to `True` if taking the mean of the errors in the loss, set to `False` if taking the sum.
        pointwise_t: Boolean variable, set to `True` if returning a function that can evaluate the loss pointwise over time. Set to `False` if returns an expectation of the loss over time.

    Returns:
        A loss function that can be used for score matching training.
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss(t, params, model, rng, data):
            n_batch = data.shape[0]
            ts = jnp.ones((n_batch,)) * t
            score = get_score(sde, model, params, score_scaling)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
                losses = losses * g2
            return jnp.mean(losses)
    else:
        def loss(params, model, rng, data):
            rng, step_rng = random.split(rng)
            n_batch = data.shape[0]
            # which one is preferable?
            # ts = random.randint(step_rng, (n_batch,), 1, solver.num_steps) / (solver.num_steps - 1)
            ts = random.uniform(step_rng, (data.shape[0],), minval=1. / solver.num_steps, maxval=1.0)
            score = get_score(sde, model, params, score_scaling)
            e = errors(ts, sde, score, rng, data, likelihood_weighting)
            losses = e**2
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
            if likelihood_weighting:
                g2 = sde.sde(jnp.zeros_like(data), ts)[1]**2
                losses = losses * g2
            return jnp.mean(losses)
    return loss

