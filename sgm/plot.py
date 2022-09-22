import jax.numpy as jnp
import jax
# enable 64 precision
# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
import jax.random as random
from functools import partial
from scipy.stats import norm
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
import matplotlib.animation as animation
from sgm.utils import drift, dispersion, train_ts, reverse_sde


BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4


# plt.rcParams.update({"text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})


def plot_samples(x, index, lims=None):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.scatter(
        x[:, index[0]], x[:, index[1]],
        color='red', label=r"$x$")
    ax.legend()
    ax.set_xlabel(r"$x_{}$".format(index[0]))
    ax.set_ylabel(r"$x_{}$".format(index[1]))
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    fig.savefig(
        "samples_x{}_x{}.png".format(index[0], index[1]),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()



def plot_video(fig, ax, animate, frames, fname, fps=20, bitrate=800, dpi=300):

    ani = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1, fargs=(ax,))
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
    # Warning: mp4 does not work on pdf
    ani.save('{}.mp4'.format(fname), writer=writer, dpi=dpi)


def plot_OH(forward_density):
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


def heatmap_image(score, N, n_samps=5000, rng=random.PRNGKey(123)):
    rng, step_rng = random.split(rng)
    samples = reverse_sde(step_rng, N, n_samps, drift, dispersion, score, train_ts)
    plot_heatmap(samples[:, [0,1]], -3, 3)
