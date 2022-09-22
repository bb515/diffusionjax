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
from scipy.stats import qmc
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
beta_max = 3

# Grid over time
R = 1000
train_ts = jnp.arange(1, R)/(R-1)

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


def sample_mvn(J, N, kernel=Linear(), m_0=0.0):
    """
    J: How many samples to generate
    N: Dimension of the embedding space
    Returns a (J, N) array of samples
    """
    rng = random.PRNGKey(123)
    rng, step_rng = jax.random.split(rng)  # the missing line
    # sample from an input space
    z = np.array(random.normal(step_rng, (N,)))
    print(N)
    C_0 = kernel(z)
    C_0 = B.dense(C_0)
    print(m_0)
    print(C_0)
    manifold = scipy.stats.multivariate_normal.rvs(mean=m_0, cov=C_0, size=J)
    # Normalization is necessary because of the unit prior distribution
    manifold = manifold - jnp.mean(manifold, axis=0)
    manifold = manifold / jnp.max(jnp.var(manifold, axis=0))
    return manifold, m_0, C_0, z


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


#we jit the function, but we have to mark some of the arguments as static,
#which means the function is recompiled every time these arguments are changed,
#since they are directly compiled into the binary code. This is necessary
#since jitted-functions cannot have functions as arguments. But it also 
#no problem since these arguments will never/rarely change in our case,
#therefore not triggering re-compilation.
# @partial(jit, static_argnums=[1, 2,3,4,5])  # removed 1 because that's N
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
