"""Utility functions related to Bayesian inversion."""

import jax.numpy as jnp
from jax import vmap, vjp, jacfwd, jacrev, grad
from diffusionjax.utils import (
  batch_mul,
  batch_matmul_A,
  batch_linalg_solve,
  batch_matmul,
  batch_mul_A,
  batch_linalg_solve_A,
)


def get_dps(sde, observation_map, y, noise_std, scale=0.4):
  """
  Implementation of score guidance suggested in
  `Diffusion Posterior Sampling for general noisy inverse problems'
  Chung et al. 2022,
  https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py

  Computes a single (batched) gradient.

  NOTE: This is not how Chung et al. 2022 implemented their method, but is a related
  continuous time method.

  Args:
    scale: Hyperparameter of the method.
      See https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
  """

  def get_l2_norm(y, estimate_h_x_0):
    def l2_norm(x, t):
      h_x_0, (s, _) = estimate_h_x_0(x, t)
      innovation = y - h_x_0
      return jnp.linalg.norm(innovation), s

    return l2_norm

  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  l2_norm = get_l2_norm(y, estimate_h_x_0)
  likelihood_score = grad(l2_norm, has_aux=True)

  def guidance_score(x, t):
    ls, s = likelihood_score(x, t)
    gs = s - scale * ls
    return gs

  return guidance_score


def get_diffusion_posterior_sampling(sde, observation_map, y, noise_std):
  """
  Implementation of score guidance suggested in
  `Diffusion Posterior Sampling for general noisy inverse problems'
  Chung et al. 2022,
  https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py
  guidance score for an observation_map that can be
  represented by either `def observation_map(x: Float[Array, dims]) -> y: Float[Array, d_x = dims.flatten()]: return mask * x  # (d_x,)`
      or `def observation_map(x: Float[Array, dims]) -> y: Float[Array, d_y]: return H @ x   # (d_y,)`
  Computes one vjps.

  NOTE: This is not how Chung et al. 2022 implemented their method, their method is `:meth:get_dps`.
  Whereas this method uses their approximation in Eq. 11 https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
  to directly calculate the score.
  """
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)

  def guidance_score(x, t):
    h_x_0, vjp_estimate_h_x_0, (s, _) = vjp(
      lambda x: estimate_h_x_0(x, t), x, has_aux=True
    )
    innovation = y - h_x_0
    C_yy = (
      noise_std**2
    )  # TODO: could investigate replacing with jnp.linalg.norm(innovation**2)
    ls = innovation / C_yy
    ls = vjp_estimate_h_x_0(ls)[0]
    gs = s + ls
    return gs

  return guidance_score


def get_pseudo_inverse_guidance(
  sde, observation_map, y, noise_std, HHT=jnp.array([1.0])
):
  """
  `Pseudo-Inverse guided diffusion models for inverse problems`
  https://openreview.net/pdf?id=9_gsMA8MRKQ
  Song et al. 2023,
  guidance score for an observation_map that can be
  represented by either `def observation_map(x: Float[Array, dims]) -> y: Float[Array, d_x = dims.flatten()]: return mask * x  # (d_x,)`
      or `def observation_map(x: Float[Array, dims]) -> y: Float[Array, d_y]: return H @ x   # (d_y,)`
  Computes one vjps.
  """
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)

  def guidance_score(x, t):
    h_x_0, vjp_estimate_h_x_0, (s, _) = vjp(
      lambda x: estimate_h_x_0(x, t), x, has_aux=True
    )
    innovation = y - h_x_0
    if HHT.shape == (y.shape[1], y.shape[1]):
      C_yy = sde.r2(t[0], data_variance=1.0) * HHT + noise_std**2 * jnp.eye(y.shape[1])
      f = batch_linalg_solve_A(C_yy, innovation)
    elif HHT.shape == (1,):
      C_yy = sde.r2(t[0], data_variance=1.0) * HHT + noise_std**2
      f = innovation / C_yy
    ls = vjp_estimate_h_x_0(f)[0]
    gs = s + ls
    return gs

  return guidance_score


def get_vjp_guidance_alt(sde, H, y, noise_std, shape):
  """
  Uses full second moment approximation of the covariance of x_0|x_t.

  Computes using H.shape[0] vjps.

  NOTE: Alternate implementation to `meth:get_vjp_guidance` that does all reshaping here.
  """
  estimate_x_0 = sde.get_estimate_x_0(lambda x: x)
  _shape = (H.shape[0],) + shape[1:]
  axes = (1, 0) + tuple(range(len(shape) + 1)[2:])
  batch_H = jnp.transpose(
    jnp.tile(H.reshape(_shape), (shape[0],) + len(shape) * (1,)), axes=axes
  )

  def guidance_score(x, t):
    x_0, vjp_x_0, (s, _) = vjp(lambda x: estimate_x_0(x, t), x, has_aux=True)
    vec_vjp_x_0 = vmap(vjp_x_0)
    H_grad_x_0 = vec_vjp_x_0(batch_H)[0]
    H_grad_x_0 = H_grad_x_0.reshape(H.shape[0], shape[0], H.shape[1])
    C_yy = sde.ratio(t[0]) * batch_matmul_A(
      H, H_grad_x_0.transpose(1, 2, 0)
    ) + noise_std**2 * jnp.eye(y.shape[1])
    innovation = y - batch_matmul_A(H, x_0.reshape(shape[0], -1))
    f = batch_linalg_solve(C_yy, innovation)
    ls = vjp_x_0(batch_matmul_A(H.T, f).reshape(shape))[0]
    gs = s + ls
    return gs

  return guidance_score


def get_vjp_guidance(sde, H, y, noise_std, shape):
  """
  Uses full second moment approximation of the covariance of x_0|x_t.

  Computes using H.shape[0] vjps.
  """
  # TODO: necessary to use shape here?
  estimate_x_0 = sde.get_estimate_x_0(lambda x: x, shape=(shape[0], -1))
  batch_H = jnp.transpose(jnp.tile(H, (shape[0], 1, 1)), axes=(1, 0, 2))
  assert y.shape[0] == shape[0]
  assert y.shape[1] == H.shape[0]

  def guidance_score(x, t):
    x_0, vjp_x_0, (s, _) = vjp(lambda x: estimate_x_0(x, t), x, has_aux=True)
    vec_vjp_x_0 = vmap(vjp_x_0)
    H_grad_x_0 = vec_vjp_x_0(batch_H)[0]
    H_grad_x_0 = H_grad_x_0.reshape(H.shape[0], shape[0], H.shape[1])
    C_yy = sde.ratio(t[0]) * batch_matmul_A(
      H, H_grad_x_0.transpose(1, 2, 0)
    ) + noise_std**2 * jnp.eye(y.shape[1])
    innovation = y - batch_matmul_A(H, x_0)
    f = batch_linalg_solve(C_yy, innovation)
    # NOTE: in some early tests it's faster to calculate via H_grad_x_0, instead of another vjp
    ls = batch_matmul(H_grad_x_0.transpose(1, 2, 0), f).reshape(s.shape)
    # ls = vjp_x_0(batch_matmul_A(H.T, f))[0]
    gs = s + ls
    return gs

  return guidance_score


def get_vjp_guidance_mask(sde, observation_map, y, noise_std):
  """
  Uses row sum of second moment approximation of the covariance of x_0|x_t.

  Computes two vjps.
  """
  # estimate_h_x_0_vmap = sde.get_estimate_x_0_vmap(observation_map)
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  batch_observation_map = vmap(observation_map)

  def guidance_score(x, t):
    h_x_0, vjp_h_x_0, (s, _) = vjp(lambda x: estimate_h_x_0(x, t), x, has_aux=True)
    diag = batch_observation_map(vjp_h_x_0(batch_observation_map(jnp.ones_like(x)))[0])
    C_yy = sde.ratio(t[0]) * diag + noise_std**2
    innovation = y - h_x_0
    ls = innovation / C_yy
    ls = vjp_h_x_0(ls)[0]
    gs = s + ls
    return gs

  return guidance_score


def get_jacrev_guidance(sde, observation_map, y, noise_std, shape):
  """
  Uses full second moment approximation of the covariance of x_0|x_t.

  Computes using d_y vjps.
  """
  batch_batch_observation_map = vmap(vmap(observation_map))
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  estimate_h_x_0_vmap = sde.get_estimate_x_0_vmap(observation_map)
  jacrev_vmap = vmap(jacrev(lambda x, t: estimate_h_x_0_vmap(x, t)[0]))

  # axes tuple for correct permutation of grad_H_x_0 array
  axes = (0,) + tuple(range(len(shape) + 1)[2:]) + (1,)

  def guidance_score(x, t):
    h_x_0, (s, _) = estimate_h_x_0(
      x, t
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jacrev_vmap(x, t)
    H_grad_H_x_0 = batch_batch_observation_map(grad_H_x_0)
    C_yy = sde.ratio(t[0]) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[1])
    innovation = y - h_x_0
    f = batch_linalg_solve(C_yy, innovation)
    ls = batch_matmul(jnp.transpose(grad_H_x_0, axes), f).reshape(s.shape)
    gs = s + ls
    return gs

  return guidance_score


def get_jacfwd_guidance(sde, observation_map, y, noise_std, shape):
  """
  Uses full second moment approximation of the covariance of x_0|x_t.

  Computes using d_y jvps.
  """
  batch_batch_observation_map = vmap(vmap(observation_map))
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  estimate_h_x_0_vmap = sde.get_estimate_x_0_vmap(observation_map)

  # axes tuple for correct permutation of grad_H_x_0 array
  axes = (0,) + tuple(range(len(shape) + 1)[2:]) + (1,)
  jacfwd_vmap = vmap(jacfwd(lambda x, t: estimate_h_x_0_vmap(x, t)[0]))

  def guidance_score(x, t):
    h_x_0, (s, _) = estimate_h_x_0(
      x, t
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    H_grad_x_0 = jacfwd_vmap(x, t)
    H_grad_H_x_0 = batch_batch_observation_map(H_grad_x_0)
    C_yy = sde.ratio(t[0]) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[1])
    innovation = y - h_x_0
    f = batch_linalg_solve(C_yy, innovation)
    ls = batch_matmul(jnp.transpose(H_grad_x_0, axes), f).reshape(s.shape)
    gs = s + ls
    return gs

  return guidance_score


def get_diag_jacrev_guidance(sde, observation_map, y, noise_std, shape):
  """Use a diagonal approximation to the variance inside the likelihood,
  This produces similar results when the covariance is approximately diagonal
  """
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  batch_batch_observation_map = vmap(vmap(observation_map))

  # axes tuple for correct permutation of grad_H_x_0 array
  axes = (0,) + tuple(range(len(shape) + 1)[2:]) + (1,)

  def vec_jacrev(x, t):
    return vmap(
      jacrev(lambda _x: estimate_h_x_0(jnp.expand_dims(_x, axis=0), t.reshape(1, 1))[0])
    )(x)

  def guidance_score(x, t):
    h_x_0, (s, _) = estimate_h_x_0(
      x, t
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    grad_H_x_0 = jnp.squeeze(vec_jacrev(x, t[0]), axis=1)
    H_grad_H_x_0 = batch_batch_observation_map(grad_H_x_0)
    C_yy = sde.ratio(t[0]) * jnp.diagonal(H_grad_H_x_0, axis1=1, axis2=2) + noise_std**2
    innovation = y - h_x_0
    f = batch_mul(innovation, 1.0 / C_yy)
    ls = batch_matmul(jnp.transpose(grad_H_x_0, axes=axes), f).reshape(s.shape)
    gs = s + ls
    return gs

  return guidance_score


def get_diag_vjp_guidance(sde, H, y, noise_std, shape):
  """
  Uses full second moment approximation of the covariance of x_0|x_t.

  Computes using H.shape[0] vjps.
  """
  # TODO: necessary to use shape here?
  estimate_x_0 = sde.get_estimate_x_0(lambda x: x, shape=(shape[0], -1))
  batch_H = jnp.transpose(jnp.tile(H, (shape[0], 1, 1)), axes=(1, 0, 2))

  def guidance_score(x, t):
    x_0, vjp_x_0, (s, _) = vjp(lambda x: estimate_x_0(x, t), x, has_aux=True)
    vec_vjp_x_0 = vmap(vjp_x_0)
    H_grad_x_0 = vec_vjp_x_0(batch_H)[0]
    H_grad_x_0 = H_grad_x_0.reshape(H.shape[0], shape[0], H.shape[1])
    diag_H_grad_H_x_0 = jnp.sum(batch_mul_A(H, H_grad_x_0.transpose(1, 0, 2)), axis=-1)
    C_yy = sde.ratio(t[0]) * diag_H_grad_H_x_0 + noise_std**2
    innovation = y - batch_matmul_A(H, x_0)
    f = batch_mul(innovation, 1.0 / C_yy)
    ls = vjp_x_0(batch_matmul_A(H.T, f))[0]
    gs = s + ls
    return gs

  return guidance_score


def get_diag_jacfwd_guidance(sde, observation_map, y, noise_std, shape):
  """Use a diagonal approximation to the variance inside the likelihood,
  This produces similar results when the covariance is approximately diagonal
  """
  batch_batch_observation_map = vmap(vmap(observation_map))
  estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
  # axes tuple for correct permutation of grad_H_x_0 array
  axes = (0,) + tuple(range(len(shape) + 1)[2:]) + (1,)

  def vec_jacfwd(x, t):
    return vmap(
      jacfwd(lambda _x: estimate_h_x_0(jnp.expand_dims(_x, axis=0), t.reshape(1, 1))[0])
    )(x)

  def guidance_score(x, t):
    h_x_0, (s, _) = estimate_h_x_0(
      x, t
    )  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
    H_grad_x_0 = jnp.squeeze(vec_jacfwd(x, t[0]), axis=(1))
    H_grad_H_x_0 = batch_batch_observation_map(H_grad_x_0)
    C_yy = sde.ratio(t[0]) * jnp.diagonal(H_grad_H_x_0, axis1=1, axis2=2) + noise_std**2
    f = batch_mul(y - h_x_0, 1.0 / C_yy)
    ls = batch_matmul(jnp.transpose(H_grad_x_0, axes=axes), f).reshape(s.shape)
    gs = s + ls
    return gs

  return guidance_score
