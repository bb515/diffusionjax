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


class NonLinear(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        in_size = x.shape[1]
        n_hidden = 256
        act = nn.relu
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=1)
        x = jnp.concatenate([x, t],axis=1)
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
    rng, step_rng = random.split(rng)
    trained_score = lambda x, t: non_linear_trained_score(score_model, params, t, x)
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


def non_linear_trained_score(score_model, params, t, x):
    v = var(t)  # (n_batch, N)
    stds = jnp.sqrt(v)
    s = score_model.apply(params, x, t)  # (n_samples, N + 1, N)
    #return s
    return s / stds


# Get a jax grad function, which can be batched with vmap
nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0, None), out_axes=(0)))
nabla_log_pt = jit(vmap(nabla_log_pt_scalar_hyperplane, in_axes=(0, 0), out_axes=(0)))
