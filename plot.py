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
import matplotlib.animation as animation


# plt.rcParams.update({"text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

rng = random.PRNGKey(123)
beta_min = 0.001
beta_max = 3

# Grid over time
R = 1000
train_ts = jnp.arange(1, R)/(R-1)



def plot_OH():
    # define data point
    x_0 = 2
    R = 10
    D = 1000
    x = jnp.linspace(-3, 3, D)
    train_ts = jnp.arange(1*R/10, R)/(R-1)
    # print(train_ts)
    Z = forward_density(x_0, x, train_ts)  # (D, R)
    # Z = forward_potential(x, train_ts)
    plt.contourf(train_ts, x, Z, 200, cmap="viridis")
    plt.savefig("contour.png")
    plt.close()
    return 0


@partial(jax.jit)
def matrix_cho(matrix):
    L_cov = cholesky(matrix, lower=True)
    return L_cov


# def error_scores(mf, m_0, C_0, area_min=-1, area_max=1):
#     """
#     Returns 
#     """
#     D = 16
#     x = jnp.linspace(area_min, area_max, D)
#     x, y = jnp.meshgrid(x, x)
#     grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
#     t = jnp.ones((grid.shape[0], 1)) * t
#     errors = jnp.empty(D, R)
#     x = jnp.linspace(-3, 3, D)
#     for i, t in enumerate(train_ts):
#         # Need a cholesky for each t so do loop
#         L_cov, mean = S_given_t(mf, t, m_0, C_0)
#         errors[:, i] = error_score_given_t(mf, x, t, L_cov, mean)
#     return errors


# def error_scores_vmap(mf, m_0, C_0, area_min=-1, area_max=1):
#     """
#     Returns 
#     """
#     D = 16
#     x = jnp.linspace(area_min, area_max, D)
#     x, y = jnp.meshgrid(x, x)
#     grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
#     t = jnp.ones((grid.shape[0], 1)) * t
#     x = jnp.linspace(-3, 3, D)
#     return error_score(mf, x, t, m_0, C_0) 


def plot_score(score, t, N, area_min=-1, area_max=1, fname="plot_score"):
    if N != 2:
        raise ValueError("WARNING: This function expects the score to be a function R² -> R²")
    #this helper function is here so that we can jit it.
    #We can not jit the whole function since plt.quiver cannot
    #be jitted
    @partial(jit, static_argnums=[0,])
    def helper(score, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0], 1)) * t
        scores = score(grid, t)
        return grid, scores
    grid, scores = helper(score, t, area_min, area_max)
    plt.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])
    plt.savefig(fname)
    plt.close()


def plot_score_ax(ax, score, t, N, area_min=-1, area_max=1, fname="plot_score"):
    if N != 2:
        raise ValueError("WARNING: This function expects the score to be a function R² -> R²")
    #this helper function is here so that we can jit it.
    #We can not jit the whole function since plt.quiver cannot
    #be jitted
    @partial(jit, static_argnums=[0,])
    def helper(score, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0], 1)) * t
        scores = score(grid, t)
        return grid, scores
    grid, scores = helper(score, t, area_min, area_max)
    ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])


def plot_score_diff(ax, score1, score2, t, N, area_min=-1, area_max=1, fname="plot_score"):
    if N != 2:
        raise ValueError("WARNING: This function expects the score to be a function R² -> R²")
    #this helper function is here so that we can jit it.
    #We can not jit the whole function since plt.quiver cannot
    #be jitted
    def helper(score1, score2, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0], 1)) * t
        scores1 = score1(grid, t)
        scores2 = score2(grid, t)
        return grid, scores1, scores2
    grid, scores1, scores2 = helper(score1, score2, t, area_min, area_max)
    norm1 = jnp.linalg.norm(scores1)
    norm2 = jnp.linalg.norm(scores2)
    diff = scores1/norm1 - scores2/norm2
    #ax.contourf(grid, jnp.linalg.norm(diff, axis=1))
    ax.quiver(grid[:, 0], grid[:, 1], diff[:, 0], diff[:, 1], angles='xy', scale_units='xy', scale=0.05)



#we jit the function, but we have to mark some of the arguments as static,
#which means the function is recompiled every time these arguments are changed,
#since they are directly compiled into the binary code. This is necessary
#since jitted-functions cannot have functions as arguments. But it also 
#no problem since these arguments will never/rarely change in our case,
#therefore not triggering re-compilation.
# @partial(jit, static_argnums=[1, 2,3,4,5])  # removed 1 because that's N
def reverse_sde(rng, N, n_samples, forward_drift, dispersion, score, ts=train_ts):
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


def plot_heatmap(positions, area_min=-3, area_max=3):
    """
    positions: locations of all particles in R^2, array (J, 2)
    area_min: lowest x and y coordinate
    area_max: highest x and y coordinate
 
    will plot a heatmap of all particles in the area [area_min, area_max] x [area_min, area_max]
    """
    def small_kernel(z, area_min, area_max):
        a = jnp.linspace(area_min, area_max, 512)
        x, y = jnp.meshgrid(a, a)
        dist = (x - z[0])**2 + (y - z[1])**2
        hm = jnp.exp(-350*dist)
        return hm

    #again we try to jit most of the code, but use the helper functions
    #since we cannot jit all of it because of the plt functions
    @jit
    def produce_heatmap(positions, area_min, area_max):
        return jnp.sum(vmap(small_kernel, in_axes=(0, None, None))(positions, area_min, area_max), axis=0)

    hm = produce_heatmap(positions, area_min, area_max) #np.sum(vmap(small_kernel)(to_plot), axis=0)
    extent = [area_min, area_max, area_max, area_min]
    plt.imshow(hm, cmap=cm, interpolation='nearest', extent=extent)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig("plot_heatmap.png")
    plt.close()


def heatmap_image(score, n_samps=5000, rng=random.PRNGKey(123)):
    rng, step_rng = random.split(rng)
    samples = reverse_sde(step_rng, N, n_samps, drift, dispersion, score)
    plot_heatmap(samples[:, [0,1]], -3, 3)



#Initialize the optimizer
optimizer = optax.adam(1e-3)

