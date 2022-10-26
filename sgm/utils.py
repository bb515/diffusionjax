import jax.numpy as jnp
import jax
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from jax.experimental.host_callback import id_print
import matplotlib.pyplot as plt
from jax.lax import scan
from jax import grad, jit, vmap
import jax.random as random
from functools import partial
from scipy.stats import norm
# from scipy.stats import qmc
from jax.scipy.special import logsumexp
from jax.scipy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
import flax.linen as nn
import optax
import scipy
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
import numpy as np  # for plotting
# For sampling from MVN
from mlkernels import Linear, EQ
import lab as B

rng = random.PRNGKey(123)
beta_min = 0.001
# TODO: beta_max usually set to 2.0?
beta_max = 3

# Grid over time
R = 1000

# Could this be a nonlinear grid over time?
train_ts = jnp.linspace(0, 1, R + 1)[1:]
# train_ts = jnp.logspace(-4, 0, R)

#Initialize the optimizer
optimizer = optax.adam(1e-3)


@partial(jax.jit, static_argnames=['N'])  # TODO: keep this here?
def matrix_inverse(matrix, N):
    L_cov = cholesky(matrix, lower=True)
    L_covT_inv = solve_triangular(L_cov, B.eye(N), lower=True)
    cov = solve_triangular(L_cov.T, L_covT_inv, lower=False)
    return cov, L_cov


@partial(jax.jit)
def matrix_cho(matrix):
    L_cov = cholesky(matrix, lower=True)
    return L_cov


@partial(jax.jit)
def matrix_solve(L_cov, b):
    x = solve_triangular(
        L_cov, b, lower=True)
    return x


def orthogonal_projection_matrix(tangent):
    return 1./ jnp.linalg.norm(tangent) * jnp.array([[tangent[0]**2, tangent[0] * tangent[1]], [tangent[0] * tangent[1], tangent[1]**2]])


def sample_mvn(J, N, kernel=Linear(), m_0=0.0):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    # rng = random.PRNGKey(123)
    # rng, step_rng = jax.random.split(rng)  # the missing line
    # sample from an input space
    # z = np.array(random.normal(step_rng, (N,)))
    z = np.linspace(-3, 3, N)
    C_0 = kernel(z)
    C_0 = B.dense(C_0)
    manifold = scipy.stats.multivariate_normal.rvs(mean=m_0, cov=C_0, size=J)
    # Normalization may be necessary because of the unit prior distribution, but must unnormalize later
    #manifold = manifold - jnp.mean(manifold, axis=0)
    #manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    marginal_mean = jnp.mean(jnp.mean(manifold, axis=0))
    manifold = manifold - marginal_mean
    marginal_std = jnp.max(jnp.std(manifold, axis=0))
    manifold = manifold / marginal_std
    return manifold, m_0, C_0, z, marginal_mean, marginal_std


def sample_sphere(J, M, N):
    """
    J: How many samples to generate
    M: Dimension of the submanifold
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    assert M <= N-1
    #Implement for other dimensions than two
    dist = scipy.stats.qmc.MultivariateNormalQMC(mean=jnp.zeros(M+1))
    manifold = jnp.array(dist.random(J))
    norms = jnp.sqrt(jnp.sum(manifold**2, axis=1).reshape((J, 1)))
    manifold = manifold/norms
    norms = jnp.sum(manifold**2, axis=1).reshape((J, 1))
    if M+1 < N:
        manifold = jnp.concatenate([manifold, jnp.zeros((J, N-(M+1)))], axis=1)
    #normalization
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold


def sample_hyperplane(J, M, N):
    """
    J: How many samples to generate
    M: Dimension of the submanifold
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    assert M <= N
    dist = scipy.stats.qmc.MultivariateNormalQMC(mean=jnp.zeros(M))  # is this better than other samplers
    manifold = jnp.array(dist.random(J))
    if M < N:
        manifold = jnp.concatenate([manifold, jnp.zeros((J, N-(M)))], axis=1)
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold


def sample_hyperplane_mvn(J, N, C_0, m_0, projection_matrix):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)  # the missing line
    # sample from an input space
    manifold = scipy.stats.multivariate_normal.rvs(mean=m_0, cov=C_0, size=J)
    # Normalization is done in practice because of the unit prior distribution
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    # Project onto the manifold
    manifold = manifold @ projection_matrix.T
    return manifold


def sample_multimodal_mvn(J, N, C_0, m_0, weights):
    """
    J is approx number of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)
    nmodes = jnp.shape(m_0)[0]
    manifolds = []
    for i in range(nmodes):
        print(m_0[i])
        print(C_0[i])
        print(int(J * weights[i]))
        manifolds.append(scipy.stats.multivariate_normal.rvs(mean=m_0[i], cov=C_0[i], size=int(J * weights[i])))
    manifold = jnp.concatenate(manifolds, axis=0)
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold


def sample_multimodal_hyperplane_mvn(J, N, C_0, m_0, weights, projection_matrix):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)  # the missing line
    nmodes = jnp.shape(m_0)[0]
    manifolds = []
    for i in range(nmodes):
        print(m_0[i])
        print(C_0[i])
        print(int(J * weights[i]))
        manifolds.append(scipy.stats.multivariate_normal.rvs(mean=m_0[i], cov=C_0[i], size=int(J * weights[i])))
    manifold = jnp.concatenate(manifolds, axis=0)
    print(jnp.shape(manifold))
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    # print(projection_matrix)
    manifold = manifold @ projection_matrix.T  # transpose since it is right multiplied
    return manifold


def beta_t(t):
    """
    t: time (number)
    returns beta_t as explained above
    """
    return beta_min + t * (beta_max - beta_min)


def alpha_t(t):
    """
    t: time (number)
    returns alpha_t as explained above
    """
    # integral of beta_t dt up to a constant, 0
    return t * beta_min + 0.5 * t**2 * (beta_max - beta_min)


def drift(x, t):
    """
    x: location of J particles in N dimensions, shape (J, N)
    t: time (number)
    returns the drift of a time-changed OU-process for each batch member, shape (J, N)
    """
    _beta = beta_t(t)
    return - 0.5 * _beta * x


def dispersion(t):
    """
    t: time (number)
    returns the dispersion
    """
    _beta = beta_t(t)
    return jnp.sqrt(_beta)


def mean_factor(t):
    """
    t: time (number)
    returns m_t as above
    """
    _alpha_t = alpha_t(t)
    return jnp.exp(-0.5 * _alpha_t)


def var(t):
    """
    t: time (number)
    returns v_t as above
    """
    _alpha_t = alpha_t(t)
    return 1.0 - jnp.exp(-_alpha_t)


def forward_potential(x_0, x, t):
    # evaluate density of marginal for a single datapoint on x_0 = [0]^T
    mean_coeff = mean_factor(t)
    v = var(t)
    x = x.reshape(-1, 1)
    density = (x - mean_coeff * x_0) / v
    return density


def forward_density(x_0, x, t):
    mean_coeff = mean_factor(t)
    v = var(t)
    x = x.reshape(-1, 1)
    return norm.pdf(x, loc = mean_coeff * x_0, scale=jnp.sqrt(v))


def forward_sde_hyperplane_t(t, rng, N_samps, m_0, C_0):
    rng, step_rng = jax.random.split(rng)
    z = random.normal(step_rng, (m_0.shape[0], N_samps))
    m_t = mean_factor(t)
    m_0 = m_0.reshape(-1, 1)
    v_t = var(t)
    C = m_t**2 * C_0 + v_t * jnp.eye(jnp.shape(m_0)[0])
    L_cov = matrix_cho(C)
    return m_t * m_0 + L_cov @ z


def average_distance_sde_hyperplane(t, m_0, C_0):
    # Is an expectation of nonlinear function - MC required
    pass


@jit
def average_distance_to_hyperplane(samples):
    J = samples.shape[0]
    return jnp.sqrt(1/J * jnp.sum(samples[:, 1:]**2))


def average_distance_to_training_data(mf, samples):
    def one_sample_distance(s, mf):
        dists = jnp.sqrt(jnp.sum((s - mf)**2, axis=1))
        return jnp.min(dists)
    
    all_dists = vmap(one_sample_distance, in_axes=(0, None), out_axes=(0))(samples, mf)
    return jnp.mean(all_dists)


def w1_dd(samples_1, samples_2):
    return scipy.stats.wasserstein_distance(samples_1, samples_2)


def w1_stdnormal(samples):
    J = samples.shape[0]
    true_inv_cdf = jax.scipy.stats.norm.ppf(jnp.linspace(0, 1, J)[1:-1])
    approx_inv_cdf = jnp.sort(samples)[1:-1]
    return jnp.mean(jnp.abs(true_inv_cdf - approx_inv_cdf))


def reverse_sde_outer(rng, N, n_samples, forward_drift, dispersion, score, ts, indices):
    xs = jnp.empty((jnp.size(indices), n_samples, N))
    j = 0
    for i in indices:
        train_ts = ts[:i]
        x = reverse_sde(rng, N, n_samples, forward_drift, dispersion, score, train_ts)
        xs = xs.at[j, :, :].set(x)
        j += 1
    return xs  # (size(indices), n_samples, N)


#we jit the function, but we have to mark some of the arguments as static,
#which means the function is recompiled every time these arguments are changed,
#since they are directly compiled into the binary code. This is necessary
#since jitted-functions cannot have functions as arguments. But it also 
#no problem since these arguments will never/rarely change in our case,
#therefore not triggering re-compilation.
@partial(jit, static_argnums=[1, 2,3,4,5])  # removed 1 because that's N
def reverse_sde_t(rng, N, n_samples, forward_drift, dispersion, score, ts):
    """
    rng: random number generator (JAX rng)
    D: dimension in which the reverse SDE runs
    N_initial: How many samples from the initial distribution N(0, I), number
    forward_drift: drift function of the forward SDE (we implemented it above)
    disperion: dispersion function of the forward SDE (we implemented it above)
    score: The score function to use as additional drift in the reverse SDE
    ts: a discretization {t_i} of [0, T], shape 1d-array
    """
    def f(carry, params):
        # drift 
        t, dt = params
        x, xs, i, rng = carry
        i += 1
        rng, step_rng = jax.random.split(rng)  # the missing line
        noise = random.normal(step_rng, x.shape)
        _dispersion = dispersion(1 - t) # jnp.sqrt(beta_{t})
        t = jnp.ones((x.shape[0], 1)) * t
        drift = -forward_drift(x, 1 - t) + _dispersion**2 * score(x, 1 - t)
        x = x + dt * drift + jnp.sqrt(dt) * _dispersion * noise
        xs = xs.at[i, :, :].set(x)
        return (x, xs, i, rng), ()
    rng, step_rng = random.split(rng)
    initial = random.normal(step_rng, (n_samples, N))
    dts = ts[1:] - ts[:-1]
    params = jnp.stack([ts[:-1], dts], axis=1)
    xs = jnp.empty((jnp.size(ts), n_samples, N))
    (_, xs, _, _), _ = scan(f, (initial, xs, 0, rng), params)
    return xs


def forward_sde_t(initial, rng, N, n_samples, forward_drift, dispersion, ts):
    """
    rng: random number generator (JAX rng)
    D: dimension in which the reverse SDE runs
    N_initial: How many samples from the initial distribution N(0, I), number
    forward_drift: drift function of the forward SDE (we implemented it above)
    disperion: dispersion function of the forward SDE (we implemented it above)
    score: The score function to use as additional drift in the reverse SDE
    ts: a discretization {t_i} of [0, T], shape 1d-array
    """
    def f(carry, params):
        # drift 
        t, dt = params
        x, xs, i, rng = carry
        rng, step_rng = jax.random.split(rng)  # the missing line
        noise = random.normal(step_rng, x.shape)
        t = jnp.ones((x.shape[0], 1)) * t
        drift = forward_drift(x, t)
        x = x + dt * drift + jnp.sqrt(dt) * dispersion(t) * noise
        xs = xs.at[i, :, :].set(x)
        i += 1
        return (x, xs, i, rng), ()
    dts = ts[1:] - ts[:-1]
    params = jnp.stack([ts[:-1], dts], axis=1)
    xs = jnp.empty((jnp.size(ts), n_samples, N))
    (_, xs, i, _), _ = scan(f, (initial, xs, 0, rng), params)
    return xs, i


#we jit the function, but we have to mark some of the arguments as static,
#which means the function is recompiled every time these arguments are changed,
#since they are directly compiled into the binary code. This is necessary
#since jitted-functions cannot have functions as arguments. But it also 
#no problem since these arguments will never/rarely change in our case,
#therefore not triggering re-compilation.
@partial(jit, static_argnums=[1, 2, 3, 4, 5])
def reverse_sde(rng, N, n_samples, forward_drift, dispersion, score, ts):
    """
    rng: random number generator (JAX rng)
    D: dimension in which the reverse SDE runs
    N_initial: How many samples from the initial distribution N(0, I), number
    forward_drift: drift function of the forward SDE (we implemented it above)
    disperion: dispersion function of the forward SDE (we implemented it above)
    score: The score function to use as additional drift in the reverse SDE
    ts: a discretization {t_i} of [0, T], shape 1d-array
    """
    def f(carry, params):
        # drift 
        t, dt = params
        x, rng = carry
        rng, step_rng = jax.random.split(rng)  # the missing line
        noise = random.normal(step_rng, x.shape)
        _dispersion = dispersion(1 - t) # jnp.sqrt(beta_{t})
        t = jnp.ones((x.shape[0], 1)) * t
        drift = -forward_drift(x, 1 - t) + _dispersion**2 * score(x, 1 - t)
        x = x + dt * drift + jnp.sqrt(dt) * _dispersion * noise
        return (x, rng), ()
    rng, step_rng = random.split(rng)
    initial = random.normal(step_rng, (n_samples, N))
    dts = ts[1:] - ts[:-1]
    params = jnp.stack([ts[:-1], dts], axis=1)
    (x, _), _ = scan(f, (initial, rng), params)
    return x


def retrain_nn_alt(update_step, N_epochs, rng, mf, score_model, params, opt_state, loss_fn, score, batch_size=5, decomposition=False):
    if decomposition:
        L = 2
    else:
        L = 1
    train_size = mf.shape[0]
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    mean_losses = jnp.zeros((N_epochs, L))
    for i in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = jnp.zeros((jnp.shape(perms)[0], L))
        for j, perm in enumerate(perms):
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn, score, has_aux=decomposition)
            if decomposition:
                loss = loss[1]
            losses = losses.at[j].set(loss)
        # TODO: is mean_loss really what I need to plot? Loses the covariance between different directions
        # For now, don't batch by setting train_size == batch_size == mf.shape[0]
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            if L==1: print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
            if L==2: print("Tangent loss {:.2f}, perpendicular loss {:.2f}".format(mean_loss[0], mean_loss[1]))
    return score_model, params, opt_state, mean_losses


def retrain_nn(
        update_step, N_epochs, rng, mf, score_model, params,
        opt_state, loss_fn, batch_size=5, decomposition=False):
    if decomposition:
        L = 2
    else:
        L = 1
    train_size = mf.shape[0]
    batch_size = min(train_size, batch_size)
    steps_per_epoch = train_size // batch_size
    mean_losses = jnp.zeros((N_epochs, L))
    for i in range(N_epochs):
        rng, step_rng = random.split(rng)
        perms = jax.random.permutation(step_rng, train_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        losses = jnp.zeros((jnp.shape(perms)[0], L))
        for j, perm in enumerate(perms):
            batch = mf[perm, :]
            rng, step_rng = random.split(rng)
            loss, params, opt_state = update_step(params, step_rng, batch, opt_state, score_model, loss_fn, has_aux=decomposition)
            if decomposition:
                loss = loss[1]
            losses = losses.at[j].set(loss)
        # TODO: is mean_loss really what I need to plot? Loses the covariance between different directions
        # For now, don't batch by setting train_size == batch_size == mf.shape[0]
        mean_loss = jnp.mean(losses, axis=0)
        mean_losses = mean_losses.at[i].set(mean_loss)
        if i % 10 == 0:
            if L==1: print("Epoch {:d}, Loss {:.2f} ".format(i, mean_loss[0]))
            if L==2: print("Tangent loss {:.2f}, perpendicular loss {:.2f}".format(mean_loss[0], mean_loss[1]))
    return score_model, params, opt_state, mean_losses


def forward_marginals(rng, time_samples, batch):
    mean_coeff = mean_factor(time_samples)
    v = var(time_samples)
    std = jnp.sqrt(v)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    return rng, mean_coeff, std, mean_coeff * batch + stds * noise


# TODO: make it hard to jit compile with likelihood flag
def flipped_errors(params, model, score, rng, N, n_batch, likelihood_flag=0):
    """
    backwards loss, not differentiating through SDE solver. Just taking samples from it,
    but need to evaluate the exact score via a large sum.
    Likely to be slow
    """
    if likelihood_flag==0:
        # Song's likelihood rescaling
        # model evaluate is h = -\sigma_t s(x_t)
        trained_score = lambda x, t: -model.evaluate(params, x, t) / jnp.sqrt(var(t))
        rescaled_score = lambda x, t: -model.evaluate(params, x, t)
    elif likelihood_flag==1:
        # What has worked previously for us, which learns a score
        trained_score = lambda x, t: model.evaluate(params, x, t)
        rescaled_score = lambda x, t: model.evaluate(params, x, t)
    elif likelihood_flag==2:
        # Jakiw training objective - has incorrect
        trained_score = lambda x, t: model.evaluate(params, x, t)
        rescaled_score = lambda x, t: model.evaluate(params, x, t)
    elif likelihood_flag==3:
        # Not likelihood rescaling
        # model evaluate is s(x_t) errors are then scaled by \beta_t
        trained_score = lambda x, t: model.evaluate(params, x, t)
        rescaled_score = lambda x, t: model.evaluate(params, x, t)
    rng, step_rng = random.split(rng)
    thinning = False
    if thinning is True:
        # Test size for last X_t on sample path
        test_size = int(5.0**N)
        indices = random.randint(step_rng, (n_batch,), 1, R)
        samples = reverse_sde_outer(rng, N, test_size, drift,
                                    dispersion, trained_score, train_ts, indices)  # (size(indices), test_size, N)
        ts = train_ts[indices]
    else:
        # Test size for keeping all X from sample path
        # Differnce is 10 times speed up in loss, not IID->introduces bias?
        # TODO: do I need an adjoint for the loss to save memory?
        test_size = int(5.0**N)
        samples = reverse_sde_t(rng, N, test_size, drift, dispersion, trained_score, train_ts)  # (R, test_size, N)
        ts = train_ts
        indices = jnp.arange(0, R, dtype=int)
    # Reshape
    ts = ts.reshape(-1, 1)
    ts = jnp.tile(ts, test_size)
    ts = ts.reshape(-1, 1)
    samples = samples.reshape(-1, samples.shape[2])
    test_rescaling = 0
    if test_rescaling:
        # Probably not justified
        return var(ts) * (trained_score(samples, ts) - score(samples, ts))
    else:
        # Doesn't seem to minimize score error well
        return trained_score(samples, ts) - score(samples, ts)
# return rescaled_score(samples, ts) + jnp.sqrt(var(ts)) * score(samples, ts)
# return rescaled_score(samples, ts) - jnp.sqrt(var(ts)) * score(samples, ts)
# return rescaled_score(samples, ts) - var(ts) * score(samples, ts)


def moving_average(a, n=100) :
    a = np.asarray(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_errors(params, model, score, rng, N, n_batch, fpath=None, likelihood_flag=0):
    """
    backwards loss, not differentiating through SDE solver. Just taking samples from it,
    but need to evaluate the exact score via a large sum.
    Likely to be slow
    """
    rng, step_rng = random.split(rng)
    #~
    if likelihood_flag==0:
        # Song's likelihood rescaling
        # model evaluate is h = -\sigma_t s(x_t)
        # Standard training objective without likelihood rescaling
        trained_score = lambda x, t: -model.evaluate(params, x, t) / jnp.sqrt(var(t))
        rescaled_score = lambda x, t: -model.evaluate(params, x, t)
    elif likelihood_flag==1:
        # What has worked previously for us, which learns a score
        trained_score = lambda x, t: model.evaluate(params, x, t)
        rescaled_score = lambda x, t: model.evaluate(params, x, t)
    elif likelihood_flag==2:
        # Jakiw training objective - has incorrect
        trained_score = lambda x, t: model.evaluate(params, x, t)
        rescaled_score = lambda x, t: model.evaluate(params, x, t)
    elif likelihood_flag==3:
        # Not likelihood rescaling
        # model evaluate is s(x_t) errors are then scaled by \beta_t
        trained_score = lambda x, t: model.evaluate(params, x, t)
        rescaled_score = lambda x, t: model.evaluate(params, x, t)

    thinning = False
    if thinning is True:
        # Test size fo X from sample path
        test_size = int(5.0**N)
        indices = random.randint(step_rng, (n_batch,), 1, R)
        samples = reverse_sde_outer(rng, N, test_size, drift,
                                    dispersion, trained_score, train_ts, indices)  # (size(indices), test_size, N)
        ts = train_ts[indices]
    else:
        # Test size for keeping all X from sample path
        # Differnce is 10 times speed up in loss, not IID->introduces bias?
        # TODO: do I need an adjoint for the loss to save memory?
        test_size = int(5.0**N)
        samples = reverse_sde_t(rng, N, test_size, drift, dispersion, trained_score, train_ts)  # (R, test_size, N)
        ts = train_ts
        indices = jnp.arange(0, R, dtype=int)
    # Reshape
    ts = ts.reshape(-1, 1)
    ts = jnp.tile(ts, test_size)
    ts = ts.reshape(-1, 1)
    indices = indices.reshape(-1, 1)
    indices = jnp.tile(indices, test_size)
    indices= indices.reshape(-1, 1).flatten()
    samples = samples.reshape(-1, samples.shape[2])
    q_score = trained_score(samples, ts)
    p_score = score(samples, ts)
    print("HERE")
    id_print(q_score)
    id_print(p_score)
    print("THERE")
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('viridis', n_batch)    # n_batch discrete colors
    colors = cmap(jnp.arange(0, R)/R)
    plt.scatter(samples[:, 0], samples[:, 1], c=colors[indices])
    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.savefig(fpath + "samples_over_t.png")
    plt.close()
    plt.scatter(q_score[:, 0], p_score[:, 0], label="0")
    plt.scatter(q_score[:, 1], p_score[:, 1], label="1")
    plt.xlim((-20.0, 20.0))
    plt.ylim((-20.0, 20.0))
    plt.savefig(fpath + "q_p.png")
    plt.close()
    errors = jnp.sum((q_score - p_score)**2, axis=1)
    plt.scatter(ts, errors.reshape(-1, 1), c=colors[indices])
    plt.savefig(fpath + "error_t.png")
    plt.close()
    # Experiment with likelihood rescaling
    test_rescaling = 0
    if test_rescaling:
        # Probably not justified
        return var(ts) * (trained_score(samples, ts) - score(samples, ts))
    else:
        # Doesn't seem to minimize score error well
        return trained_score(samples, ts) - score(samples, ts)


def errors(params, model, rng, batch, likelihood_flag=0):
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
    if likelihood_flag==0:
        # Song's likelihood rescaling
        # model evaluate is h = \sigma_t s(x_t)
        # Standard training objective without likelihood rescaling
        return noise - model.evaluate(params, x_t, time_samples)
    elif likelihood_flag==1:
        # What has worked previously for us, which learns a score
        # It seems to be best to scale by stds - implies learnign actual loss
        return noise + stds * model.evaluate(params, x_t, time_samples)
    elif likelihood_flag==2:
        # Jakiw training objective - has incorrect
        return noise + v * model.evaluate(params, x_t, time_samples)
    elif likelihood_flag==3:
        # Not likelihood rescaling
        # model evaluate is s(x_t) errors are then scaled by \beta_t
        return (noise / stds + model.evaluate(params, x_t, time_samples)) * dispersion(time_samples)


def errors_t(t, params, model, rng, batch):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    n_batch = batch.shape[0]
    times = jnp.ones((n_batch, 1)) * t
    mean_coeff = mean_factor(times)  # (n_batch, N)
    v = var(times)  # (n_batch, N)
    stds = jnp.sqrt(v)
    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)
    x_t = mean_coeff * batch + stds * noise # (n_batch, N)
    return noise + model.evaluate(params, x_t, times)


def flipped_loss_fn(params, model, score, rng, mf):
    (n_batch, N) = jnp.shape(mf)
    e = flipped_errors(params, model, score, rng, N, n_batch)
    return jnp.mean(jnp.sum(e**2, axis=1))


def loss_fn(params, model, rng, batch):
    e = errors(params, model, rng, batch)
    return jnp.mean(jnp.sum(e**2, axis=1))
    # TODO: option for likelihood rescaling
    # TODO: should be a lambda function of some variables?


def loss_fn_t(t, params, model, rng, batch):
    e = errors_t(t, params, model, rng, batch)
    # TODO: option for likelihood rescaling
    # TODO: should be a lambda function of some variables?
    return jnp.mean(jnp.sum(e**2, axis=1))


def orthogonal_loss_fn_t(projection_matrix):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    def decomposition(t, params, model, rng, batch, projection_matrix):
        rng, step_rng = random.split(rng)
        n_batch = batch.shape[0]
        e = errors_t(t, params, model, rng, batch)
        loss = jnp.mean(jnp.sum(e**2, axis=1))
        parallel = jnp.mean(jnp.sum(jnp.dot(e, projection_matrix.T)**2, axis=1))
        perpendicular = loss - parallel
        return loss, jnp.array([parallel, perpendicular]) 
    return lambda t, params, model, rng, batch: decomposition(t, params, model, rng, batch, projection_matrix)


def orthogonal_loss_fn(projection_matrix):
    """
    params: the current weights of the model
    model: the score function
    rng: random number generator from jax
    batch: a batch of samples from the training data, representing samples from \mu_text{data}, shape (J, N)
    
    returns an random (MC) approximation to the loss \bar{L} explained above
    """
    def decomposition(params, model, rng, batch, projection_matrix):
        rng, step_rng = random.split(rng)
        n_batch = batch.shape[0]
        time_samples = random.randint(step_rng, (n_batch, 1), 1, R) / (R - 1)  # why these not independent? I guess that they can be? (n_samps,)
        e = errors(params, model, rng, batch)
        loss = jnp.mean(jnp.sum(e**2, axis=1))
        parallel = jnp.mean(jnp.sum(jnp.dot(e, projection_matrix.T)**2, axis=1))
        perpendicular = loss - parallel
        return loss, jnp.array([parallel, perpendicular]) 
    return lambda params, model, rng, batch: decomposition(params, model, rng, batch, projection_matrix)


@partial(jit, static_argnums=[4, 5, 6, 7])
# TODO work out workaround for it not being possible to jit dynamic indexing
# of the sample time
def reverse_update_step(params, rng, batch, opt_state, model, loss_fn, score, has_aux=False):
    # TODO: There isn't a more efficient place to factorize jax value_and_grad?
    val, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, model, score, rng, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return val, params, opt_state


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


def get_mf(data_string, Js, J_true, M, N):
    """Get the manifold data."""
    # TODO: try a 2-D or M-D basis 
    tangent_basis = 3.0 * jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2)])
    # Tangent vector needs to have unit norm
    tangent_basis /= jnp.linalg.norm(tangent_basis)
    # tangent_basis = jnp.array([1.0, 0.1])
    projection_matrix = orthogonal_projection_matrix(tangent_basis)
    # Note so far that tangent_basis only implemented for 1D basis
    # tangent_basis is dotted with (N, n_batch) errors, so must be (N, 1)
    tangent_basis = tangent_basis.reshape(-1, 1)
    print(tangent_basis)
    print(projection_matrix)
    if data_string=="hyperplane":
        # For 1D hyperplane example,
        C_0 = jnp.array([[1, 0], [0, 0]])
        m_0 = jnp.zeros(N)
        mf_true = sample_hyperplane(J_true, M, N)
    elif data_string=="hyperplane_mvn":
        mf_true = sample_hyperplane_mvn(J_true, N, C_0, m_0, projection_matrix)
        C_0 = jnp.array([[1, 0], [0, 0]])
        m_0 = jnp.zeros(N)
    elif data_string=="multimodal_hyperplane_mvn":
        # For 1D multimodal hyperplane example,
        m_0 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        C_0 = jnp.array(
            [
                [[0.05, 0.0], [0.0, 0.1]],
                [[0.05, 0.0], [0.0, 0.1]]
            ]
        )
        weights = jnp.array([0.5, 0.5])
        N = 100
        mf_true = sample_multimodal_hyperplane_mvn(J_true, N, C_0, m_0, weights, projection_matrix)
    elif data_string=="multimodal_mvn":
        mf_true = sample_multimodal_mvn(J, N, C_0, m_0, weights)
    elif data_string=="sample_sphere":
        m_0 = None
        C_0 = None
        mf_true = sample_sphere(J_true, M, N)
    else:
        raise NotImplementedError()
    mfs = {}
    for J in Js:
        mfs["{:d}".format(J)] = mf_true[:J, :]
    return mfs, mf_true, m_0, C_0, tangent_basis, projection_matrix

