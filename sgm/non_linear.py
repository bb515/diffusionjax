import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import jax.random as random
from jax.scipy.special import logsumexp
from jax import vmap, jit, grad
from sgm.utils import (
    matrix_solve, matrix_cho,
    optimizer)
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


def log_hat_pt(sde, x, t, mf):
    """
    Empirical distribution score for normal distribution on the hyperplane.

    x: One location in R^n, N should be 2 in special case example
    t: time
    """
    N = mf.shape[0]
    mean, std = sde.marginal_prob(x, t)
    potentials = jnp.sum(-(x - mean)**2 / (2 * std**2), axis=1)
    return logsumexp(potentials, axis=0, b=1/N)


def nabla_log_pt_scalar(sde, x, t, m_0, C_0):
    """
    Analytical distribution score for normal distribution on the hyperplane.
    Requires a linear solve for each x
    Requires a cholesky for each t

    x: One location in R^n, N should be 2 in special case example.
    t: time
    """
    m_t = sde.mean_coeff(t)
    N = m_0.shape[0]
    v_t = sde.variance(t)
    mean = m_t * m_0
    mat = m_t**2 * C_0 + v_t
    if N == 1:
        return (x - mean) / mat
    else:
        L_cov = matrix_cho(mat)
        return -matrix_solve(L_cov, x - mean)


def nabla_log_pt_scalar_hyperplane(sde, x, t):
    N = x.shape[0]
    m_t = sde.mean_coeff(t)
    v_t = sde.variance(t)
    mat = m_t**2 * jnp.array([[1, 0], [0, 0]]) + v_t * jnp.identity(N)
    if N == 1:
        return (x) / mat
    else:
        L_cov = matrix_cho(mat)
        return -matrix_solve(L_cov, x)


def non_linear_trained_score(score_model, params, t, x):
    v = var(t)  # (n_batch, N)
    stds = jnp.sqrt(v)
    s = score_model.apply(params, x, t)  # (n_samples, N + 1, N)
    return s / stds


# Get a jax grad function, which can be batched with vmap
nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0, None), out_axes=(0)))
nabla_log_pt = jit(vmap(nabla_log_pt_scalar_hyperplane, in_axes=(0, 0), out_axes=(0)))
