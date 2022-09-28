import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import jax.random as random
from jax import vmap, jit
from sgm.utils import (
    mean_factor, var, R,
    matrix_inverse, matrix_solve,
    optimizer)
from functools import partial
from jax.experimental.host_callback import id_print
from sgm.non_linear import update_step as nonlinear_update_step


# def S_given_t(mf, t, m_0, C_0):
#     N = mf.shape[0]
#     mean_coeff = mean_factor(t)
#     mean = m_0 * mean_coeff
#     v = var(t)
#     L_cov = matrix_cho(mean_coeff**2 * C_0 + v)
#     return L_cov, mean


# def log_pt_factored_t(x, L_cov, mean):
#     """
#     Analytical distribution  for normal distribution on the hyperplane.
#     Requires a linear solve for each x
#     Requires a cholesky for each t

#     x: One location in R^n, N should be 2 in special case example.
#     t: time    
#     """
#     return -matrix_solve(L_cov, x - mean)


class ApproximateScoreLinear(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        batch_size = x.shape[0]
        N = jnp.shape(x)[1]
        in_size = (N + 1) * N
        n_hidden = 256
        s = jnp.array([t - 0.5, jnp.cos(2*jnp.pi*t)]).T
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(in_size)(s)
        s = jnp.reshape(s, (batch_size, N + 1, N))  # (batch_size, N + 1, N)
        s = jnp.einsum('ijk, ij -> ik', s, jnp.hstack((jnp.ones((batch_size, 1)), x)))
        return s  # (n_batch,)

    # def evaluate_eig(self, x, t):
    #     batch_size = x.shape[0]
    #     N = jnp.shape(x)[1]
    #     in_size = (N + 1) * N
    #     n_hidden = 256
    #     s = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=1)
    #     s = nn.Dense(n_hidden)(s)
    #     s = nn.relu(s)
    #     s = nn.Dense(n_hidden)(s)
    #     s = nn.relu(s)
    #     s = nn.Dense(n_hidden)(s)
    #     s = nn.relu(s)
    #     s = nn.Dense(in_size)(s)
    #     s = jnp.reshape(s, (batch_size, N + 1, N))  # (batch_size, N + 1, N)
    #     #print(s[0, :, 1])
    #     #print(s[0, :, 1:])
    #     x = jnp.einsum('ijk, ij -> ik', s, jnp.hstack((jnp.ones((batch_size, 1)), x)))
    #     print(s[0, 0, :])
    #     print(jnp.linalg.eig(s[0, 1:, :]))
    #     print(jnp.hstack((jnp.ones((batch_size, 1)), x)))


class ApproximateScoreOperatorLinear(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""
    # Not sure how to isolate the matrix, S, and have it included in a loss? Just write the loss in terms of the batches.

    @nn.compact
    def __call__(self, t, N):
        batch_size = jnp.size(t)
        in_size = (N + 1) * N
        n_hidden = 256
        # s = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=1)  # (n_batch, 2)
        s = jnp.array([t - 0.5, jnp.cos(2*jnp.pi*t)]).T  # (n_batch, 2)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(n_hidden)(s)
        s = nn.relu(s)
        s = nn.Dense(in_size)(s)
        s = jnp.reshape(s, (batch_size, N + 1, N))  # (batch_size, N + 1, N)
        return s


def linear_loss_fn(params, model, rng, batch):
    """
    params: the current weights of the model
    model: the  function
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
    N = jnp.shape(x_t)[1]
    s = model.apply(params, x_t, time_samples)  #  (n_batch, N)
    return jnp.mean(jnp.sum((noise + s  * stds)**2, axis=1))


def loss_fn(params, model, rng, batch):
    """
    params: the current weights of the model
    model: the  function
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
    N = jnp.shape(x_t)[1]
    s = model.apply(params, time_samples, N)  #  (n_batch, N, N+1) which is quite memory intense
    s = jnp.einsum('ijk, ij -> ik', s, jnp.hstack((jnp.ones((n_batch, 1)), x_t)))
    return jnp.mean(jnp.sum((noise + s  * stds)**2, axis=1))


def loss_fn_t(t, params, model, rng, batch):
    """
    params: the current weights of the model
    model: the  function
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
    N = jnp.shape(x_t)[1]
    s = model.apply(params, times, N)  #  (n_batch, N, N+1) which is quite memory intense
    s = jnp.einsum('ijk, ij -> ik', s, jnp.hstack((jnp.ones((n_batch, 1)), x_t)))
    return jnp.mean(jnp.sum((noise + s  * stds)**2, axis=1))


def true_loss_fn_t(params, model, t, m_0, C_0):
    """
    Analytical distribution  for normal distribution on the hyperplane.
    Requires a linear solve for each x
    Requires a cholesky for each t

    x: One location in R^n, N should be 2 in special case example.
    t: time    
    """
    N = jnp.shape(m_0)[0]
    v_t = var(t)  # (N,)
    m_t = mean_factor(t)
    mat = m_t**2 * C_0 + v_t
    mean = m_0 * m_t
    s = model.apply(params, t, N)  # (N, N+1) which is memory intense
    loss, residual, intermediate = true_loss_scalar(s, mat, mean, N)
    return loss * v_t


def true_loss_scalar(s, mat, mean, N):
    if N == 1:
        # mat_inv = 1./ mat
        # L_mat = jnp.sqrt(mat)
        # return (x - m_0) / mat
        raise ValueError("Not implemented")
    else:
        mat_inv, _ = matrix_inverse(mat, N)
    # It might be s[:, 0] and s[:, 1:]
    S = s[:, 1:][0]
    mu = s[:, 0][0]
    residual = - S @ mean - mu
    intermediate = S @ mat + jnp.eye(N)
    residual_loss = residual.T @ residual
    trace_loss = jnp.einsum('ij, ji -> ', intermediate, S + mat_inv)
    loss = residual_loss + trace_loss
    return loss, residual, intermediate


def true_loss_fn(params, model, rng, n_batch, m_0, C_0):
    """
    Analytical distribution  for normal distribution on the hyperplane.
    Requires a linear solve for each x
    Requires a cholesky for each t

    x: One location in R^n, N should be 2 in special case example.
    t: time
    """
    N = jnp.shape(m_0)[0]
    rng, step_rng = random.split(rng)
    time_samples = random.randint(step_rng, (n_batch, 1), 1, R) / (R - 1)
    v_t = var(time_samples)  # (n_batch, N)
    m_t = mean_factor(time_samples)[0]  # (n_batch, N)
    mat = m_t**2 * C_0 + v_t  # (n_batch, N, N+1)
    mean = m_0 * m_t  # (n_batch, N)
    s = model.apply(params, time_samples, N)  #  (n_batch, N, N+1) which is quite memory intense
    loss = true_loss_scalar(s, mat, mean, N)
    return loss


def train_linear_nn(rng, mf, batch_size, score_model, N_epochs):
    rng, step_rng = random.split(rng)
    train_size = mf.shape[0]
    N = mf.shape[1]
    batch_size = 5
    batch_size = min(train_size, batch_size)
    x = jnp.zeros(N*batch_size).reshape((batch_size, N))
    time = jnp.ones((batch_size, 1))
    params = score_model.init(step_rng, x, time)
    opt_state = optimizer.init(params)
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
            loss, params, opt_state = nonlinear_update_step(params, step_rng, batch, opt_state, score_model, linear_loss_fn)
            losses.append(loss)
        mean_loss = jnp.mean(jnp.array(losses))
        if k % 1 == 0:
            print("Epoch %d \t, Loss %f " % (k, mean_loss))
    return score_model, params


# @partial(jit, static_argnums=[6, 7, 8])
def update_step(n_batch, m_0, C_0, params, rng, opt_state, model, loss_fn, has_aux=False):
    """
    params: the current weights of the model
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    opt_state: the internal state of the optimizer
    model: the  function

    takes the gradient of the loss function and updates the model weights (params) using it. Returns
    the value of the loss function (for metrics), the new params and the new optimizer state
    """
    # TODO: There isn't a more efficient place to factorize jax value_and_grad?
    val, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, model, rng, n_batch, m_0, C_0)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


def retrain_nn(
        n_batch, m_0, C_0, N_epochs, rng,
        _model, params, opt_state, loss_fn,
        decomposition=False):
    if decomposition:
        L = 2
    else:
        L = 1
    steps_per_epoch = 1
    print("N_epochs = {}, steps_per_epoch = {}".format(N_epochs, steps_per_epoch))
    mean_losses = jnp.zeros((N_epochs, L))
    for i in range(N_epochs):
        rng, step_rng = random.split(rng)
        loss, params, opt_state = update_step(
            n_batch, m_0, C_0, params, step_rng,
            opt_state, _model, loss_fn,
            has_aux=decomposition)
        if decomposition:
            loss = loss[1]
        mean_losses = mean_losses.at[i].set(loss)
        if i % 1 == 0:
            if L==1: print(
                "Epoch {:d}, Loss {:.2f} ".format(i, loss))
            if L==2: print(
                "Tangent loss {:.2f}, perpendicular loss {:.2f}".format(loss[0], loss[1]))
    return _model, params, opt_state, mean_losses
