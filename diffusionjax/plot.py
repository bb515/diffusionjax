"""Helper functions for plots in the example."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap
from functools import partial
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)
import matplotlib.animation as animation

BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4


def plot_heatmap(samples, area_min=-3, area_max=3, fname="plot_heatmap"):
    """Plots a heatmap of all samples in the area [area_min, area_max] x [area_min, area_max].
    Args:
        samples: locations of all particles in R^2, array (J, 2)
        area_min: lowest x and y coordinate
        area_max: highest x and y coordinate
    """
    def small_kernel(z, area_min, area_max):
        a = jnp.linspace(area_min, area_max, 512)
        x, y = jnp.meshgrid(a, a)
        dist = (x - z[0])**2 + (y - z[1])**2
        hm = jnp.exp(-350 * dist)
        return hm

    # We try to jit most of the code, but use the helper functions
    # since we cannot jit all of it because of the plt functions
    @jit
    def produce_heatmap(samples, area_min, area_max):
        return jnp.sum(vmap(small_kernel, in_axes=(0, None, None))(samples, area_min, area_max), axis=0)

    hm = produce_heatmap(samples, area_min, area_max)
    extent = [area_min, area_max, area_max, area_min]
    plt.imshow(hm, cmap=cm, interpolation='nearest', extent=extent)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig(fname)
    plt.close()


def plot_samples(samples, index, fname="samples.png", lims=None):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.scatter(
        samples[:, index[0]], samples[:, index[1]],
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
        fname,
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


def plot_animation(fig, ax, animate, frames, fname, fps=20, bitrate=800, dpi=300):

    ani = animation.FuncAnimation(
        fig, animate, frames=frames, interval=1, fargs=(ax,))
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
    # Note that mp4 does not work on pdf
    ani.save('{}.mp4'.format(fname), writer=writer, dpi=dpi)


def plot_score(score, t, area_min=-1, area_max=1, fname="plot_score"):
    fig, ax = plt.subplots(1, 1)
    # this helper function is here so that we can jit it.
    # We can not jit the whole function since plt.quiver cannot
    # be jitted
    @partial(jit, static_argnums=[0,])
    def helper(score, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0],)) * t
        scores = score(grid, t)
        return grid, scores
    grid, scores = helper(score, t, area_min, area_max)
    ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    plt.gca().set_aspect('equal', adjustable='box')
    fig.savefig(fname)
    plt.close()


def plot_score_ax(ax, score, t, area_min=-1, area_max=1):
    # This helper function is here so that we can jit it.
    # We can not jit the whole function since plt.quiver cannot be jitted
    @partial(jit, static_argnums=[0,])
    def helper(score, t, area_min, area_max):
        x = jnp.linspace(area_min, area_max, 16)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        t = jnp.ones((grid.shape[0],)) * t
        scores = score(grid, t)
        return grid, scores
    grid, scores = helper(score, t, area_min, area_max)
    ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 3])
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")


def plot_heatmap_ax(ax, samples, area_min=-3, area_max=3):
    """Plots a heatmap of all samples in the area [area_min, area_max] x [area_min, area_max].
    Args:
        samples: locations of all particles in R^2, array (J, 2)
        area_min: lowest x and y coordinate
        area_max: highest x and y coordinate
    """
    def small_kernel(z, area_min, area_max):
        a = jnp.linspace(area_min, area_max, 512)
        x, y = jnp.meshgrid(a, a)
        dist = (x - z[0])**2 + (y - z[1])**2
        hm = jnp.exp(-350 * dist)
        return hm

    # We try to jit most of the code, but use the helper functions
    # since we cannot jit all of it because of the plt functions
    @jit
    def produce_heatmap(samples, area_min, area_max):
        return jnp.sum(vmap(small_kernel, in_axes=(0, None, None))(samples, area_min, area_max), axis=0)

    hm = produce_heatmap(samples, area_min, area_max)
    extent = [area_min, area_max, area_max, area_min]
    ax.imshow(hm, cmap=cm, interpolation='nearest', extent=extent)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")


def plot_temperature_schedule(sde, solver):
    """Plots the temperature schedule of the SDE marginals.

    Args:
        sde: a valid SDE class.
    """
    m2 = sde.mean_coeff(solver.ts)
    v = sde.variance(solver.ts)
    plt.plot(solver.ts, m2, label="m2")
    plt.plot(solver.ts, v, label="v")
    plt.legend()
    plt.savefig("plot_temperature_schedule.png")
    plt.close()


def plot_scatter(samples, fname="samples"):
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.scatter(
        samples[:, 0], samples[:, 1],
        alpha=0.1, label=r"$x$")
    ax.legend()
    ax.set_xlabel(r"$x_{}$".format(0))
    ax.set_ylabel(r"$x_{}$".format(1))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    fig.savefig(
        fname,
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
