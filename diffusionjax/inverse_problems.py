"""Utility functions related to Bayesian inversion."""
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, vjp
import jax.random as random
from diffusionjax.utils import batch_mul


def get_pseudo_inverse_guidance_mask(
        sde, observation_map, shape, y, noise_std):
    """
    `Pseudo-Inverse guided diffusion models for inverse problems`
    https://openreview.net/pdf?id=9_gsMA8MRKQ
    Song et al. 2023,
    guidance score for an observation_map that can be
    represented by a `def observation_map(x): return mask * x`.
    Computes one vjps.
    """
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def guidance_score(x, t):
        h_x_0, vjp_estimate_h_x_0, (s, x_0) = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = sde.r2(t[0], data_variance=1.) + noise_std**2
        f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)[0]
        gs = s + ls
        return gs

    return guidance_score


def get_vjp_guidance_mask(
        sde, observation_map, shape, y, noise_std):
    """
    Uses row sum of second moment approximation of the covariance of x_0|x_t.

    Computes two vjps.
    """
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def guidance_score(x, t):
        h_x_0, vjp_h_x_0, (s, _) = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        diag = observation_map(vjp_h_x_0(observation_map(jnp.ones(x.shape)))[0])
        C_yy = sde.ratio(t[0]) * diag + noise_std**2
        innovation = y - h_x_0
        ls = innovation / C_yy
        ls = vjp_h_x_0(ls)[0]
        gs = s + ls
        return gs

    return guidance_score


def get_diffusion_posterior_sampling_mask(
        sde, observation_map, shape, y, noise_std):
    """
    Implementation of score guidance suggested in
    `Diffusion Posterior Sampling for general noisy inverse problems'
    Chung et al. 2022,
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py
    guidance score for an observation_map that can be
    represented by a `def observation_map(x): return mask * x`.
    Computes one vjps.

    NOTE: This is not how Chung et al. 2022 implemented their method, their method is `:meth:get_dps_mask`.
    Whereas this method uses their approximation in Eq. 11 https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
    to directly calculate the score.
    """
    estimate_h_x_0 = sde.get_estimate_x_0(observation_map)
    def guidance_score(x, t):
        h_x_0, vjp_estimate_h_x_0, (s, x_0) = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = noise_std**2
        ls = innovation / C_yy
        ls = vjp_estimate_h_x_0(ls)[0]
        gs = s + ls
        return gs

    return guidance_score
