import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import jax.random as random
from jax.scipy.special import logsumexp
from jax import vmap, jit, grad
from sgm.utils import (
    mean_factor, var, R,
    matrix_solve, matrix_cho,
    optimizer, reverse_sde,
    drift, dispersion, train_ts)
from functools import partial


class ApproximateScore(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        in_size = x.shape[1]
        n_hidden = 256
        act = nn.relu
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)],axis=1)
        x = jnp.concatenate([x, t],axis=1)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x


def log_hat_pt(x, t, mf):
    """
    Empirical distribution score for normal distribution on the hyperplane.

    x: One location in R^n, N should be 2 in special case example
    t: time
    """
    N = mf.shape[0]
    means = mf * mean_factor(t)
    v = var(t)
    potentials = jnp.sum(-(x - means)**2 / (2 * v), axis=1)  # doesn't potential also depends on jnp.log(_v_t)? but taking nabla_x
    return logsumexp(potentials, axis=0, b=1/N)


def nabla_log_pt_scalar(x, t, m_0, C_0):
    """
    Analytical distribution score for normal distribution on the hyperplane.
    Requires a linear solve for each x
    Requires a cholesky for each t

    x: One location in R^n, N should be 2 in special case example.
    t: time    
    """
    N = m_0.shape[0]
    mean_coeff = mean_factor(t)
    mean = m_0 * mean_coeff
    v = var(t)
    mat = mean_coeff**2 * C_0 + v
    if N == 1:
        return (x - m_0) / mat
    else:
        L_cov = matrix_cho(mean_coeff**2 * C_0 + v)
        return -matrix_solve(L_cov, x - mean)


def loss_fn(params, model, rng, batch):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    rng, step_rng = random.split(rng)
    n_batch = batch.shape[0]
    time_samples = random.randint(step_rng, (n_batch, 1), 1, R) / (R - 1)  # why these not independent? I guess that they can be? (n_samps,)
    mean_coeff = mean_factor(time_samples)  # (n_batch, N)
    v = var(time_samples)  # (n_batch, N)
    stds = jnp.sqrt(v)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean_coeff * batch + stds * noise # (n_batch, N)
    s = model.apply(params, x_t, time_samples)
    return jnp.mean(jnp.sum((noise + s * stds)**2, axis=1))  # TODO: maybe there is a mistake here? should it be scaled by root(v) instead?


def loss_fn_t(t, params, model, rng, batch):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    rng, step_rng = random.split(rng)
    n_batch = batch.shape[0]
    times = jnp.ones((n_batch, 1)) * t
    mean_coeff = mean_factor(times)  # (n_batch, N)
    v = var(times)  # (n_batch, N)
    stds = jnp.sqrt(v)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean_coeff * batch + stds * noise # (n_batch, N)
    s = model.apply(params, x_t, times)
    return jnp.mean(jnp.sum((noise + s * stds)**2, axis=1))


def orthogonal_loss_fn_t(tangent_basis):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """

    def loss_fn_t(t, params, model, rng, batch, tangent_basis):
        """
        params: the current weights of the model
        model: the score function
        rng: random number generator from jax
        batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
        
        returns an random (MC) approximation to the loss \bar{L} explained above
        """
        rng, step_rng = random.split(rng)
        n_batch = batch.shape[0]
        times = jnp.ones((n_batch, 1)) * t
        mean_coeff = mean_factor(times)  # (n_batch, N)
        v = var(times)  # (n_batch, N)
        stds = jnp.sqrt(v)
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, batch.shape)
        x_t = mean_coeff * batch + stds * noise # (n_batch, N)
        s = model.apply(params, x_t, times)
         # find orthogonal components of the loss
        loss = jnp.mean(jnp.sum((noise + s * stds)**2, axis=1))
        parallel = jnp.mean(jnp.sum(jnp.dot(noise + s * stds, tangent_basis)**2, axis=1))
        perpendicular = loss - parallel
        return loss, jnp.array([parallel, perpendicular])

    return lambda t, params, model, rng, batch: loss_fn_t(t, params, model, rng, batch, tangent_basis)


def orthogonal_loss_fn(tangent_basis):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    def loss_fn(params, model, rng, batch, tangent_basis):
        rng, step_rng = random.split(rng)
        n_batch = batch.shape[0]
        time_samples = random.randint(step_rng, (n_batch, 1), 1, R) / (R - 1)  # why these not independent? I guess that they can be? (n_samps,)
        mean_coeff = mean_factor(time_samples)  # (n_batch, N)
        v = var(time_samples)  # (n_batch, N)
        stds = jnp.sqrt(v)
        rng, step_rng = random.split(rng)
        noise = random.normal(step_rng, batch.shape)
        x_t = mean_coeff * batch + stds * noise # (n_batch, N)
        s = model.apply(params, x_t, time_samples)  # (n_batch, N)
        # find orthogonal components of the loss
        loss = jnp.mean(jnp.sum((noise + s * stds)**2, axis=1))
        parallel = jnp.mean(jnp.sum(jnp.dot(noise + s * stds, tangent_basis)**2, axis=1))
        perpendicular = loss - parallel
        return loss, jnp.array([parallel, perpendicular])
    return lambda params, model, rng, batch: loss_fn(params, model, rng, batch, tangent_basis)


@partial(jit, static_argnums=[4, 5, 6])
def update_step(params, rng, batch, opt_state, model, loss_fn, has_aux=False):
    """
    params: the current weights of the model
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    opt_state: the internal state of the optimizer
    model: the score function

    takes the gradient of the loss function and updates the model weights (params) using it. Returns
    the value of the loss function (for metrics), the new params and the new optimizer state
    """
    # TODO: There isn't a more efficient place to factorize jax value_and_grad?
    val, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, model, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


def train_nn(mf, N):
    """Train nn"""
    rng = random.PRNGKey(123)
    ## Neural network training
    batch_size = 16
    #some dummy input data. Flax is able to infer all the dimensions of the weights
    #if we supply if with the kind of input data it has to expect
    x = jnp.zeros(N*batch_size).reshape((batch_size, N))
    time = jnp.ones((batch_size, 1))
    #initialize the model weights
    score_model = ApproximateScore()
    params = score_model.init(rng, x, time)
    #Initialize the optimizer
    opt_state = optimizer.init(params)
    N_epochs = 10000
    train_size = mf.shape[0]
    batch_size = 50
    steps_per_epoch = train_size // batch_size
    for k in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = []
        for perm in perms:
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn)
            losses.append(loss)
        mean_loss = jnp.mean(jnp.array(losses))
        if k % 10 == 0:
            print("Epoch %d \t, Loss %f " % (k, mean_loss))
    trained_score = lambda x, t: score_model.apply(params, x, t)
    rng, step_rng = random.split(rng)
    return reverse_sde(step_rng, N, 1000, drift, dispersion, trained_score, train_ts)


def nabla_log_pt_scalar_hyperplane(x, t):
    N = x.shape[0]
    mean_coeff = mean_factor(t)**2
    v = var(t)
    mat = mean_coeff * jnp.array([[1, 0], [0, 0]]) + v * jnp.identity(N)
    if N == 1:
        return (x) / mat
    else:
        L_cov = matrix_cho(mat)
        return -matrix_solve(L_cov, x)


# Get a jax grad function, which can be batched with vmap
nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0, None), out_axes=(0)))
nabla_log_pt = jit(vmap(nabla_log_pt_scalar_hyperplane, in_axes=(0, 0), out_axes=(0)))
