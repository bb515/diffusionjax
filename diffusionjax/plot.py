"""Plotting code for the examples."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap
from functools import partial
import matplotlib.animation as animation
import numpy as np


BG_ALPHA = 1.0
MG_ALPHA = 1.0
FG_ALPHA = 0.3
color_posterior = "#a2c4c9"
color_algorithm = "#ff7878"
dpi_val = 1200
cmap = "magma"


def plot_heatmap(samples, area_bounds, lengthscale=350.0, fname="plot_heatmap") -> None:
  """Plots a heatmap of all samples in the area area_bounds x area_bounds.
  Args:
    samples: locations of particles shape (num_particles, 2)
  """

  def small_kernel(z, area_bounds):
    a = jnp.linspace(area_bounds[0], area_bounds[1], 512)
    x, y = jnp.meshgrid(a, a)
    dist = (x - z[0]) ** 2 + (y - z[1]) ** 2
    hm = jnp.exp(-lengthscale * dist)
    return hm

  @jit  # jit most of the code, but use the helper functions since cannot jit all of it because of plt
  def produce_heatmap(samples, area_bounds):
    return jnp.sum(vmap(small_kernel, in_axes=(0, None))(samples, area_bounds), axis=0)

  hm = produce_heatmap(samples, area_bounds)
  extent = area_bounds + area_bounds
  plt.imshow(hm, interpolation="nearest", extent=extent)
  ax = plt.gca()
  ax.invert_yaxis()
  plt.savefig(fname + ".png")
  plt.close()


def image_grid(x, image_size, num_channels):
  img = x.reshape(-1, image_size, image_size, num_channels)
  w = int(np.sqrt(img.shape[0]))
  return (
    img.reshape((w, w, image_size, image_size, num_channels))
    .transpose((0, 2, 1, 3, 4))
    .reshape((w * image_size, w * image_size, num_channels))
  )


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
  img = image_grid(x, image_size, num_channels)
  plt.figure(figsize=(8, 8))
  plt.axis("off")
  plt.imshow(img, cmap=cmap)
  plt.savefig(fname + ".png", bbox_inches="tight", pad_inches=0.0)
  # plt.savefig(fname + '.pdf', bbox_inches='tight', pad_inches=0.0)
  plt.close()


def plot_scatter(samples, index, fname="samples", lims=None):
  fig, ax = plt.subplots(1, 1)
  fig.patch.set_facecolor("white")
  fig.patch.set_alpha(BG_ALPHA)
  ax.scatter(samples[:, index[0]], samples[:, index[1]], color="red", label=r"$x$")
  ax.legend()
  ax.set_xlabel(r"$x_{}$".format(index[0]))
  ax.set_ylabel(r"$x_{}$".format(index[1]))
  if lims is not None:
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
  plt.gca().set_aspect("equal", adjustable="box")
  plt.draw()
  fig.savefig(fname + ".png", facecolor=fig.get_facecolor(), edgecolor="none")
  plt.close()


def plot_samples_1D(samples, image_size, x_max=5.0, fname="samples 1D", alpha=FG_ALPHA):
  x = np.linspace(-x_max, x_max, image_size)
  plt.plot(x, samples[..., 0].T, alpha=alpha)
  plt.xlim(-5.0, 5.0)
  plt.ylim(-5.0, 5.0)
  plt.savefig(fname + ".png")
  plt.close()


def plot_animation(fig, ax, animate, frames, fname, fps=20, bitrate=800, dpi=300):
  ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1, fargs=(ax,))
  # Set up formatting for the movie files
  Writer = animation.writers["ffmpeg"]
  writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=bitrate)
  # Note that mp4 does not work on pdf
  ani.save("{}.mp4".format(fname), writer=writer, dpi=dpi)


def plot_score(score, scaler, t, area_bounds=[-3.0, 3.0], fname="plot_score"):
  fig, ax = plt.subplots(1, 1)

  # this helper function is here so that we can jit
  @partial(
    jit,
    static_argnums=[
      0,
    ],
  )  # We can not jit the whole function since plt.quiver cannot be jitted
  def helper(score, t, area_bounds):
    x = jnp.linspace(area_bounds[0], area_bounds[1], 16)
    x, y = jnp.meshgrid(x, x)
    grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
    t = jnp.ones((grid.shape[0],)) * t
    scores = score(scaler(grid), t)
    return grid, scores

  grid, scores = helper(score, t, area_bounds)
  ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1])
  ax.set_xlabel(r"$x_0$")
  ax.set_ylabel(r"$x_1$")
  plt.gca().set_aspect("equal", adjustable="box")
  fig.savefig(fname + ".png")
  plt.close()


def plot_score_ax(ax, score, scaler, t, area_bounds=[-3.0, 3.0]):
  @partial(
    jit,
    static_argnums=[
      0,
    ],
  )  # We can not jit the whole function since plt.quiver cannot be jitted
  def helper(score, t, area_bounds):
    x = jnp.linspace(area_bounds[0], area_bounds[1], 16)
    x, y = jnp.meshgrid(x, x)
    grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
    t = jnp.ones((grid.shape[0],)) * t
    scores = score(scaler(grid), t)
    return grid, scores

  grid, scores = helper(score, t, area_bounds)
  ax.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 3])
  ax.set_xlabel(r"$x_0$")
  ax.set_ylabel(r"$x_1$")


def plot_heatmap_ax(ax, samples, area_bounds=[-3.0, 3.0], lengthscale=350):
  """Plots a heatmap of all samples in the area area_bounds^{2}.
  Args:
    samples: locations of all particles in R^2, array (J, 2)
  """

  def small_kernel(z, area_bounds):
    a = jnp.linspace(area_bounds[0], area_bounds[1], 512)
    x, y = jnp.meshgrid(a, a)
    dist = (x - z[0]) ** 2 + (y - z[1]) ** 2
    hm = jnp.exp(-lengthscale * dist)
    return hm

  @jit
  def produce_heatmap(samples, area_bounds):
    return jnp.sum(
      vmap(small_kernel, in_axes=(0, None, None))(samples, area_bounds), axis=0
    )

  hm = produce_heatmap(samples, area_bounds)
  extent = area_bounds + area_bounds
  ax.imshow(hm, interpolation="nearest", extent=extent)
  ax = plt.gca()
  ax.invert_yaxis()
  ax.set_xlabel(r"$x_0$")
  ax.set_ylabel(r"$x_1$")


def plot_temperature_schedule(sde, solver):
  """Plots the temperature schedule of the SDE marginals.

  Args:
    sde: a valid SDE class.
  """
  m2 = sde.mean_coeff(solver.ts) ** 2
  v = sde.variance(solver.ts)
  plt.plot(solver.ts, m2, label="m2")
  plt.plot(solver.ts, v, label="v")
  plt.legend()
  plt.savefig("plot_temperature_schedule.png")
  plt.close()
