"""JAX port of Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models".
Ported from the code https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


def pixel_normalize(x, channel_axis, eps=1e-4):
  """
  Normalize given tensor to unit magnitude with respect to the given
  channel axis.
  Args:
    x: Assume (N, C, H, W)
  """
  norm = jnp.linalg.vector_norm(x, axis=channel_axis, keepdims=True)
  norm = eps + jnp.sqrt(norm.size / x.size) * norm
  return x / norm


def weight_normalize(x, eps=1e-4):
  """
  Normalize given tensor to unit magnitude with respect to all the dimensions
  except the first.
  Args:
    x: Assume (N, C, H, W)
  """
  norm = jax.vmap(lambda x: jnp.linalg.vector_norm(x, keepdims=True))(x)
  norm = eps + jnp.sqrt(norm.size / x.size) * norm
  return x / norm


def resample(x, f=[1, 1], mode="keep"):
  """
  Upsample or downsample the given tensor with the given filter,
  or keep it as is.

  Args:
    x: Assume (N, C, H, W)
  """
  if mode == "keep":
    return x
  f = jnp.float32(f)
  assert f.ndim == 1 and len(f) % 2 == 0
  f = f / f.sum()
  f = jnp.outer(f, f)[jnp.newaxis, jnp.newaxis, :, :]
  c = x.shape[1]

  if mode == "down":
    return jax.lax.conv_general_dilated(
      x,
      jnp.tile(f, (c, 1, 1, 1)),
      window_strides=(2, 2),
      feature_group_count=3,
      padding="SAME",  # not sure
    )
  assert mode == "up"

  pad = (len(f) - 1) // 2 + 1
  return jax.lax.conv_general_dilated(
    x,
    jnp.tile(f * 4, (c, 1, 1, 1)),
    dimension_numbers=("NCHW", "OIHW", "NCHW"),
    window_strides=(1, 1),
    lhs_dilation=(2, 2),
    feature_group_count=3,
    padding=((pad, pad), (pad, pad)),
  )


def mp_silu(x):
  """Magnitude-preserving SiLU (Equation 81)."""
  return nn.activation.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
  """Magnitude-preserving sum (Equation 88)."""
  return (a + t * (b - a)) / jnp.sqrt((1 - t) ** 2 + t**2)


def mp_cat(a, b, dim=1, t=0.5):
  """Magnitude-preserving concatenation (Equation 103)."""
  Na = a.shape[dim]
  Nb = b.shape[dim]
  C = jnp.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
  wa = C / jnp.sqrt(Na) * (1 - t)
  wb = C / jnp.sqrt(Nb) * t
  return jax.lax.concatenate([wa * a, wb * b], dim)


class MPFourier(nn.Module):
  """Magnitude-preserving Fourier features (Equation 75)."""

  num_channels: int
  bandwidth: float = 1.0

  @nn.compact
  def __call__(self, x):
    freqs = self.param(
      "freqs",
      jax.nn.initializers.normal(stddev=2 * jnp.pi * self.bandwidth),
      (self.num_channels,),
    )
    freqs = jax.lax.stop_gradient(freqs)
    phases = self.param(
      "phases", jax.nn.initializers.normal(stddev=2 * jnp.pi), (self.num_channels,)
    )
    phases = jax.lax.stop_gradient(phases)
    y = jnp.outer(x, freqs)
    y = y + phases
    y = jnp.cos(y) * jnp.sqrt(2)
    return y


class MPConv(nn.Module):
  """Magnitude-preserving convolution or fully-connected layer (Equation 47)
  with force weight normalization (Equation 66).
  """

  in_channels: int
  out_channels: int
  kernel_shape: tuple
  training: bool = True

  @nn.compact
  def __call__(self, x, gain=1.0):
    w = self.param(
      "w",
      jax.nn.initializers.normal(stddev=1.0),
      (self.out_channels, self.in_channels, *self.kernel_shape),
    )
    if self.training:
      w = jax.lax.stop_gradient(w)
      w = weight_normalize(w)  # forced weight normalization

    print(w.shape)
    w = weight_normalize(w)  # traditional weight normalization
    w = w * (gain / jnp.sqrt(w[0].size))  # magnitude-preserving scaling
    if w.ndim == 2:
      return x @ w.T  # not sure about this
    assert w.ndim == 4

    return jax.lax.conv(
      x,
      w,
      window_strides=(1, 1),
      padding="SAME",
    )


class Block(nn.Module):
  """
  U-Net encoder/decoder block with optional self-attention (Figure 21).
  """

  in_channels: int  # Number of input channels
  out_channels: int  # Number of output channels
  emb_channels: int  # Number of embedding channels
  flavor: str = "enc"  # Flavor: 'enc' or 'dec'
  resample_mode: str = "keep"  # Resampling: 'keep', 'up', or 'down'.
  resample_filter: tuple = (1, 1)  # Resampling filter.
  attention: bool = False  # Include self-attention?
  channels_per_head: int = 64  # Number of channels per attention head.
  dropout: float or bool = 0.0  # Dropout probability.
  res_balance: float = 0.3  # Balance between main branch (0) and residual branch (1).
  attn_balance: float = 0.3  # Balance between main branch (0) and self-attention (1).
  clip_act: int = 256  # Clip output activations. None = do not clip.
  num_heads: int = out_channels // channels_per_head if attention else 0
  training: bool = True

  @nn.compact
  def __call__(self, x, emb):
    # Main branch
    x = resample(x, f=self.resample_filter, mode=self.resample_mode)
    if self.flavor == "enc":
      if self.in_channels != self.out_channels:
        x = MPConv(
          self.in_channels, self.out_channels, kernel_shape=[1, 1], name="conv_skip"
        )(x)
      x = pixel_normalize(x, channel_axis=1)  # pixel norm

    # Residual branch
    y = MPConv(
      self.out_channels if self.flavor == "enc" else self.in_channels,
      self.out_channels,
      kernel_shape=[3, 3],
      name="conv_res0",
    )(mp_silu(x))

    c = (
      MPConv(self.emb_channels, self.out_channels, kernel_shape=[], name="emb_linear")(
        emb, gain=self.param("emb_gain", jax.nn.initializers.zeros, (1,))
      )
      + 1
    )
    y = mp_silu(y * jnp.expand_dims(jnp.expand_dims(c, axis=2), axis=3))
    if self.dropout:
      y = nn.Dropout(self.dropout)(y, deterministic=not self.training)
    y = MPConv(
      self.out_channels, self.out_channels, kernel_shape=[3, 3], name="conv_res1"
    )(y)

    # Connect the branches
    if self.flavor == "dec" and self.in_channels != self.out_channels:
      x = MPConv(
        self.in_channels, self.out_channels, kernel_shape=[1, 1], name="conv_skip"
      )(x)
    x = mp_sum(x, y, t=self.res_balance)

    # Self-attention
    # TODO: test if flax.linen.attention can be used instead here
    if self.num_heads != 0:
      y = MPConv(
        self.out_channels, self.out_channels * 3, kernel_shape=[1, 1], name="attn_qkv"
      )(x)
      y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
      q, k, v = pixel_normalize(y, channel_axis=2)  # pixel normalization and split
      w = jnp.softmax(jnp.einsum("nhcq,nhck->nhqk", q, k / jnp.sqrt(q.shape[2])))
      y = jnp.einsum("nhqk,nhck->nhcq", w, v)
      y = MPConv(
        self.out_channels, self.out_channels, kernel_shape=[1, 1], name="attn_proj"
      )(y.reshape(*x.shape))
      x = mp_sum(x, y, t=self.attn_balance)

    # Clip activations
    if self.clip_act is not None:
      x = jnp.clip(x, -self.clip_act, self.clip_act)
    return x
