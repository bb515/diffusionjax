"""All functions related to loss computation and optimization."""
import flax
import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
import jax.random as random
from sgm.utils import get_score_fn, matrix_inverse, check_dims


def plot_reverse_errors(sde, score_fn, true_score_fn, rng, N, n_batch, likelihood_weighting=True):
    """
    backwards loss, not differentiating through SDE solver. Just taking samples from it,
    but need to evaluate the exact score via a large sum - will not scale to large data
    """
    rng, step_rng = random.split(rng)
    thinning = False
    if thinning is True:
        # Test size for last X_t on sample path
        # These are not sample trajectories and so may not be related to the KL divergence
        test_size = int(5.0**N)
        indices = random.randint(step_rng, (n_batch,), 1, sde.n_steps)
        samples = reverse_sde_outer(rng, N, test_size, drift,
                                    dispersion, trained_score, train_ts, indices)  # (size(indices), test_size, N)
        ts = train_ts[indices]
    else:
        # Test size for keeping all X from sample path
        # Differnce is ~10 times speed up in loss, but not IID->introduces bias? or needed to satisfy KL divergence
        # TODO: do I need an adjoint for the loss to save memory?
        test_size = int(5.0**N)
        samples = reverse_sde_t(rng, N, test_size, drift, dispersion, trained_score, train_ts)  # (sde.n_steps, test_size, N)
        ts = train_ts
        indices = jnp.arange(sde.n_steps, dtype=int)
    # Reshape
    ts = ts.reshape(-1, 1)
    ts = jnp.tile(ts, test_size)
    ts = ts.reshape(-1, 1)
    indices = indices.reshape(-1, 1)
    indices = jnp.tile(indices, test_size)
    indices= indices.reshape(-1, 1).flatten()
    samples = samples.reshape(-1, samples.shape[2])
    q = score_fn(samples, 1-ts)
    p = true_score_fn(samples, 1-ts)
    if not likelihood_weighting:
        # Does result in losses are small and do not decrease
        q = std * q
        p = std * p
    else:
        temp=0  # this is a list of alternative weightings
        if temp==0:
            # This leads to losses that are numerically very large, since
            # there is no scaling going on to weight small time
            # less than large time
            # Doesn't seem to minimize score error well
            pass
        elif temp==1:
            pass
        elif temp==2:
            # This has losses that are numerically sensible, and decrease
            q = std**2 * q
            p = std**2 * p
        elif temp==3:
            q = std * q
            p = std**2 * p
    plt.scatter(samples[:, 0], samples[:, 1], c=colors[indices])
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.savefig(fpath + "samples_over_t.png")
    plt.close()
    plt.scatter(q_score[:, 0][::-1], p_score[:, 0][::-1], c=colors[indices], label="0", alpha=0.1)
    plt.scatter(q_score[:, 1][::-1], p_score[:, 1][::-1], c=colors[indices], label="1", alpha=0.1)
    plt.xlim((-20.0, 20.0))
    plt.ylim((-20.0, 20.0))
    plt.savefig(fpath + "q_p20.png")
    plt.xlim((-200.0, 200.0))
    plt.ylim((-200.0, 200.0))
    plt.savefig(fpath + "q_p200.png")
    plt.xlim((-2000.0, 2000.0))
    plt.ylim((-2000.0, 2000.0))
    plt.savefig(fpath + "q_p2000.png")
    plt.close()
    errors = jnp.sum((q_score - p_score)**2, axis=1)
    plt.scatter(ts, errors.reshape(-1, 1), c=colors[indices])
    plt.savefig(fpath + "error_t.png")
    plt.close()
    # Experiment with likelihood rescaling
    return q - p


def errors(ts, sde, score_fn, rng, batch, likelihood_weighting=True):
    """
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    returns an random (MC) approximation to the (weighted) score errors
    """
    mean, std = sde.marginal_prob(batch, ts)  # (n_batch, N)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean + std * noise # (n_batch, N)
    if not likelihood_weighting:
        return noise - std * score_fn(x_t, ts)
    else:
        return noise / std - score_fn(x_t, ts)


def get_loss_fn(sde, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for training with arbritary SDEs.
    TODO obtions for train/evaluate and reduce_mean?
    TODO use reduce_op here
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss_fn(t, params, model, rng, batch):
            n_batch = batch.shape[0]
            ts = jnp.ones((n_batch, 1)) * t
            score_fn = get_score_fn(sde, model, params, score_scaling)
            e = errors(ts, sde, score_fn, rng, batch, likelihood_weighting)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), ts)[1]
                e = e * g
            return jnp.mean(jnp.sum(e**2, axis=1))
    else:
        def loss_fn(params, model, rng, batch):
            rng, step_rng = random.split(rng)
            n_batch = batch.shape[0]
            ts = random.randint(step_rng, (n_batch, 1), 1, sde.n_steps) / (sde.n_steps - 1)  # why these not independent? I guess that they can be? (n_samps,)
            score_fn = get_score_fn(sde, model, params, score_scaling)
            e = errors(ts, sde, score_fn, rng, batch, likelihood_weighting)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), ts)[1]
                e = e * g
            return jnp.mean(jnp.sum(e**2, axis=1))
    return loss_fn


def oracle_errors(h, mat, mean, std, N, projection_matrix=None):
    """
    arg projection_matrix: Projection matrix
    """
    if N == 1:
        # mat_inv = 1./ mat
        # L_mat = jnp.sqrt(mat)
        # return (x - m_0) / mat
        raise ValueError("Not implemented")
    else:
        mat_inv, _ = matrix_inverse(mat, N)
    mu = h[:, 0][0]
    H = h[:, 1:][0]
    residual = - (H @ mean + mu)
    intermediate = std * (H @ mat + jnp.eye(N))
    residual_loss = residual.T @ residual
    #trace_loss = jnp.trace((H @ mat + jnp.eye(N)) @ (2 * std * H).T)
    # trace_loss = std * jnp.trace(H @ mat @ H.T) + std * jnp.trace(H) + std * jnp.trace(mat_inv)
    # trace_loss = jnp.einsum('ij, ij -> ', intermediate, H + H)
    loss = residual_loss  # + trace_loss
    if projection_matrix is not None:
        projected_residual = projection_matrix @ residual
        projected_residual_loss = projected_residual.T @ projected_residual
        # projected_trace_loss = jnp.einsum('ij, ji -> ', projection_matrix @ intermediate, projection_matrix @ (H + mat_inv))
        projected = projected_residual_loss # + projected_trace_loss
        return loss, residual, intermediate, projected
    else:
        return loss, residual, intermediate


def get_oracle_loss_fn(sde, model, m_0, C_0, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False, projection_matrix=None):
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    # Check arrays are expected size
    check_dims(m_0, C_0)
    if pointwise_t:
        def loss_fn(t, params, model):
            N = jnp.shape(m_0)[0]
            m_t = sde.mean_coeff(t)
            var_t = sde.variance(t)
            std = jnp.sqrt(var_t)
            mat = (m_t**2 / std) * C_0 + std
            mean = m_0 * m_t
            h = model.apply(params, t, N)  # (N, N+1) which is memory intense
            if projection_matrix is None:
                mse, residual, intermediate = oracle_errors(h, mat, mean, std, N)
                return mse
            else:
                mse, residual, intermediate, projection = oracle_errors(h, mat, mean, std, N, projection_matrix)
                return mse, jnp.array([projection, mse - projection])
    else:
        def loss_fn(params, model, rng):
            N = jnp.shape(m_0)[0]
            rng, step_rng = random.split(rng)
            # Expectation over t
            ts = random.randint(step_rng, (n_batch, 1), 1, R) / (R - 1)
            m_t = sde.mean_coeff(ts)
            var_t = sde.variance(ts)
            std = jnp.sqrt(var_t)
            mat = (m_t**2 / std) * C_0 + std  # (n_batch, N, N+1)
            mean = m_0 * m_t  # (n_batch, N)
            h = model.apply(params, time_samples, N)  #  (n_batch, N, N+1) which is quite memory intense
            if projection_matrix is None:
                mse, residual, intermediate = oracle_errors(s, mat, mean, std, N)
                return mse
            else:
                mse, residual, intermediate, projection = oracle_errors(h, mat, mean, std, N, projection_matrix)
                return mse, jnp.array([projection, mse - projection])
    return loss_fn


def reverse_errors(sde, score_fn, true_score_fn, rng, N, n_batch, likelihood_weighting=True):
    """
    backwards loss, not differentiating through SDE solver. Just taking samples from it,
    but need to evaluate the exact score via a large sum - will not scale to large data
    """
    rng, step_rng = random.split(rng)
    thinning = False
    if thinning is True:
        # Test size for last X_t on sample path
        # These are not sample trajectories and so may not be related to the KL divergence
        test_size = int(5.0**N)
        indices = random.randint(step_rng, (n_batch,), 1, sde.n_steps)
        samples = reverse_sde_outer(rng, N, test_size, drift,
                                    dispersion, trained_score, train_ts, indices)  # (size(indices), test_size, N)
        ts = train_ts[indices]
    else:
        # Test size for keeping all X from sample path
        # Differnce is ~10 times speed up in loss, but not IID->introduces bias? or needed to satisfy KL divergence
        # TODO: do I need an adjoint for the loss to save memory?
        test_size = int(5.0**N)
        samples = reverse_sde_t(rng, N, test_size, drift, dispersion, trained_score, train_ts)  # (sde.n_steps, test_size, N)
        ts = train_ts
        indices = jnp.arange(sde.n_steps, dtype=int)
    # Reshape
    ts = ts.reshape(-1, 1)
    ts = jnp.tile(ts, test_size)
    ts = ts.reshape(-1, 1)
    samples = samples.reshape(-1, samples.shape[2])

    if not likelihood_weighting:
        # Does result in losses are small and do not decrease
        return std * (score_fn(samples, 1-ts) - true_score_fn(samples, 1-ts))
    else:
        temp=0  # this is a list of alternative weightings
        if temp==0:
            # This leads to losses that are numerically very large, since
            # there is no scaling going on to weight small time
            # less than large time
            # Doesn't seem to minimize score error well
            return score_fn(samples, 1-ts) - true_score_fn(samples, 1-ts)  # * dispersion(1-ts)
        elif temp==1:
            return (trained_score(samples, 1-ts) - true_score_fn(samples, 1-ts)) # * dispersion(1-ts)
        elif temp==2:
            # This has losses that are numerically sensible, and decrease
            return std**2 * (score_fn(samples, 1-ts) - true_score_fn(samples, 1-ts))
        elif temp==3:
            # Check the errors of the scores here. Should really plot absolute errors
            # This has losses that are numerically sensible, and decrease
            return std * score_fn(samples, 1-ts) - std**2 * true_score_fn(samples, 1-ts), ts


def get_reverse_loss_fn(sde, true_score, model, score_scaling=True, likelihood_weighting=True, pointwise_t=False):
    """Create a loss function for training with arbritary SDEs.
    TODO obtions for train/evaluate and reduce_mean?
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss_fn(t, params, model, rng, batch):
            n_batch = batch.shape[0]
            score_fn = get_score_fn(sde, model, params, score_scaling)
            ts = jnp.ones((n_batch, 1)) * t
            error = reverse_errors(ts, sde, score_fn, true_score_fn, rng, N, n_batch)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), 1-ts)[1]
                e = e * g
            return jnp.mean(jnp.sum(e**2, axis=1))
    else:
        def loss_fn(params, model, rng, batch):
            rng, step_rng = random.split(rng)
            n_batch = batch.shape[0]
            score_fn = get_score_fn(sde, model, params, score_scaling)
            ts = random.randint(step_rng, (n_batch, 1), 1, sde.n_steps) / (sde.n_steps - 1)  # why these not independent? I guess that they can be? (n_samps,)
            error = reverse_errors(ts, sde, score_fn, true_score_fn, rng, N, n_batch, likelihood_weighting)
            if likelihood_weighting:
                diffusion = sde.sde(jnp.zeros_like(data), 1-ts)[1]
                error = error * g
            return jnp.mean(jnp.sum(e**2, axis=1))
    return loss_fn


def get_projected_loss_fn(projection_matrix, sde, model, score_scaling=True, likelihood_weighting=True, reduce_mean=True, pointwise_t=False):
    """Create a loss function for training with arbritary SDEs.
    TODO obtions for train/evaluate and reduce_mean?
    """
    reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    if pointwise_t:
        def loss_fn(t, params, model, rng, batch):
            n_batch = batch.shape[0]
            score_fn = get_score_fn(sde, model, params, score_scaling)
            ts = jnp.ones((n_batch, 1)) * t
            rng, step_rng = random.split(rng)
            n_batch = batch.shape[0]
            e = errors(ts, sde, score_fn, rng, batch, likelihood_weighting)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), ts)[1]
                e = e * g
            loss = jnp.mean(jnp.sum(e**2, axis=1))
            parallel = jnp.mean(jnp.sum(jnp.dot(e, projection_matrix.T)**2, axis=1))
            perpendicular = loss - parallel
            return loss, jnp.array([parallel, perpendicular])
    else:
        def loss_fn(params, model, rng, batch):
            n_batch = batch.shape[0]
            score_fn = get_score_fn(sde, model, params, score_scaling)
            rng, step_rng = random.split(rng)
            ts = random.randint(step_rng, (n_batch, 1), 1, sde.n_steps) / (sde.n_steps - 1)  # why these not independent? I guess that they can be? (n_samps,)
            e = errors(ts, sde, score_fn, rng, batch, likelihood_weighting)
            if likelihood_weighting:
                g = sde.sde(jnp.zeros_like(batch), ts)[1]
                e = e * g
            loss = jnp.mean(jnp.sum(e**2, axis=1))
            parallel = jnp.mean(jnp.sum(jnp.dot(e, projection_matrix.T)**2, axis=1))
            perpendicular = loss - parallel
            return loss, jnp.array([parallel, perpendicular])
    return loss_fn
