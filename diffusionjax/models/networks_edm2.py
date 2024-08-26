"""JAX port of Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models".
Ported from the code https://github.com/NVlabs/edm2/blob/main/training/networks_edm2.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any


def jax_unstack(x, axis=0):
  """https://github.com/google/jax/discussions/11028"""
  return [
    jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])
  ]


def pixel_normalize(x, channel_axis, eps=1e-4):
  """
  Normalize given tensor to unit magnitude with respect to the given
  channel axis.
  Args:
    x: Assume (N, C, H, W)
  """
  norm = jnp.float32(jnp.linalg.vector_norm(x, axis=channel_axis, keepdims=True))
  norm = eps + jnp.sqrt(norm.size / x.size) * norm
  return x / jnp.array(norm, dtype=x.dtype)


def weight_normalize(x, eps=1e-4):
  """
  Normalize given tensor to unit magnitude with respect to all the dimensions
  except the first.
  Args:
    x: Assume (N, C, H, W)
  """
  norm = jnp.float32(jax.vmap(lambda x: jnp.linalg.vector_norm(x, keepdims=True))(x))
  norm = eps + jnp.sqrt(norm.size / x.size) * norm
  return x / jnp.array(norm, dtype=x.dtype)


def forced_weight_normalize(x, eps=1e-4):
  """
  Normalize given tensor to unit magnitude with respect to all the dimensions
  except the first. Don't take gradients through the computation.
  Args:
    x: Assume (N, C, H, W)
  """
  norm = jax.lax.stop_gradient(
    jnp.float32(jax.vmap(lambda x: jnp.linalg.vector_norm(x, keepdims=True))(x))
  )
  norm = eps + jnp.sqrt(norm.size / x.size) * norm
  return x / jnp.array(norm, dtype=x.dtype)


def resample(x, f=[1, 1], mode="keep"):
  """
  Upsample or downsample the given tensor with the given filter,
  or keep it as is.

  Args:
    x: Assume (N, C, H, W)
  """
  if mode == "keep":
    return x
  f = jnp.array(f, dtype=x.dtype)
  assert f.ndim == 1 and len(f) % 2 == 0
  f = f / f.sum()
  f = jnp.outer(f, f)[jnp.newaxis, jnp.newaxis, :, :]
  c = x.shape[1]

  if mode == "down":
    return jax.lax.conv_general_dilated(
      x,
      jnp.tile(f, (c, 1, 1, 1)),
      window_strides=(2, 2),
      feature_group_count=c,
      padding="SAME",
    )
  assert mode == "up"

  pad = (len(f) - 1) // 2 + 1
  return jax.lax.conv_general_dilated(
    x,
    jnp.tile(f * 4, (c, 1, 1, 1)),
    dimension_numbers=("NCHW", "OIHW", "NCHW"),
    window_strides=(1, 1),
    lhs_dilation=(2, 2),
    feature_group_count=c,
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
  return jax.lax.concatenate([wa * a, wb * b], dimension=dim)


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
    y = jnp.float32(x)
    y = jnp.float32(jnp.outer(x, freqs))
    y = y + jnp.float32(phases)
    y = jnp.cos(y) * jnp.sqrt(2)
    return jnp.array(y, dtype=x.dtype)


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
    w = jnp.float32(
      self.param(
        "w",
        jax.nn.initializers.normal(stddev=1.0),
        (self.out_channels, self.in_channels, *self.kernel_shape),
      )
    )  # TODO: type promotion required in JAX?
    if self.training:
      w = forced_weight_normalize(w)  # forced weight normalization

    w = weight_normalize(w)  # traditional weight normalization
    w = w * (gain / jnp.sqrt(w[0].size))  # magnitude-preserving scaling
    w = jnp.array(w, dtype=x.dtype)
    if w.ndim == 2:
      return x @ w.T
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
  dropout: float = 0.0  # Dropout probability.
  res_balance: float = 0.3  # Balance between main branch (0) and residual branch (1).
  attn_balance: float = 0.3  # Balance between main branch (0) and self-attention (1).
  clip_act: int = 256  # Clip output activations. None = do not clip.
  training: bool = True

  @nn.compact
  def __call__(self, x, emb):
    # Main branch
    x = resample(x, f=self.resample_filter, mode=self.resample_mode)
    if self.flavor == "enc":
      if self.in_channels != self.out_channels:
        x = MPConv(
          self.in_channels, self.out_channels, kernel_shape=(1, 1), name="conv_skip"
        )(x)
      x = pixel_normalize(x, channel_axis=1)  # pixel norm

    # Residual branch
    y = MPConv(
      self.out_channels if self.flavor == "enc" else self.in_channels,
      self.out_channels,
      kernel_shape=(3, 3),
      name="conv_res0",
    )(mp_silu(x))

    c = (
      MPConv(self.emb_channels, self.out_channels, kernel_shape=(), name="emb_linear")(
        emb, gain=self.param("emb_gain", jax.nn.initializers.zeros, (1,))
      )
      + 1
    )
    y = jnp.array(
      mp_silu(y * jnp.expand_dims(jnp.expand_dims(c, axis=2), axis=3)), dtype=y.dtype
    )
    if self.dropout:
      y = nn.Dropout(self.dropout)(y, deterministic=not self.training)
    y = MPConv(
      self.out_channels, self.out_channels, kernel_shape=(3, 3), name="conv_res1"
    )(y)

    # Connect the branches
    if self.flavor == "dec" and self.in_channels != self.out_channels:
      x = MPConv(
        self.in_channels, self.out_channels, kernel_shape=(1, 1), name="conv_skip"
      )(x)
    x = mp_sum(x, y, t=self.res_balance)

    # Self-attention
    # TODO: test if flax.linen.SelfAttention can be used instead here?
    num_heads = self.out_channels // self.channels_per_head if self.attention else 0
    if num_heads != 0:
      y = MPConv(
        self.out_channels, self.out_channels * 3, kernel_shape=(1, 1), name="attn_qkv"
      )(x)
      y = y.reshape(y.shape[0], num_heads, -1, 3, y.shape[2] * y.shape[3])
      q, k, v = jax_unstack(
        pixel_normalize(y, channel_axis=2), axis=3
      )  # pixel normalization and split
      # NOTE: quadratic cost in last dimension
      w = nn.softmax(jnp.einsum("nhcq,nhck->nhqk", q, k / jnp.sqrt(q.shape[2])), axis=3)
      y = jnp.einsum("nhqk,nhck->nhcq", w, v)
      y = MPConv(
        self.out_channels, self.out_channels, kernel_shape=(1, 1), name="attn_proj"
      )(y.reshape(*x.shape))
      x = mp_sum(x, y, t=self.attn_balance)

    # Clip activations
    if self.clip_act is not None:
      x = jnp.clip(x, -self.clip_act, self.clip_act)
    return x


class UNet(nn.Module):
  """EDM2 U-Net model (Figure 21)."""

  img_resolution: int  # Image resolution.
  img_channels: int  # Image channels.
  label_dim: int  # Class label dimensionality. 0 = unconditional.
  model_channels: int = 192  # Base multiplier for the number of channels.
  channel_mult: tuple = (
    1,
    2,
    3,
    4,
  )  # Per-resolution multipliers for the number of channels.
  channel_mult_noise: Any = None  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
  channel_mult_emb: Any = None  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
  num_blocks: int = 3  # Number of residual blocks per resolution.
  attn_resolutions: tuple = (16, 8)  # List of resolutions with self-attention.
  label_balance: float = (
    0.5  # Balance between noise embedding (0) and class embedding (1).
  )
  concat_balance: float = 0.5  # Balance between skip connections (0) and main path (1).

  # **block_kwargs - arguments for Block
  resample_filter: tuple = (1, 1)  # Resampling filter
  channels_per_head: int = 64  # Number of channels per attention head
  dropout: float = 0.0  # Dropout probability
  res_balance: float = 0.3  # Balance between main branch (0) and residual branch (1)
  attn_balance: float = 0.3  # Balance between main branch (0) and self-attention (1)
  clip_act: int = 256  # Clip output activations. None = do not clip
  out_gain: Any = None
  block_kwargs = {
    "resample_filter": resample_filter,
    "channels_per_head": channels_per_head,
    "dropout": dropout,
    "res_balance": res_balance,
    "attn_balance": attn_balance,
    "clip_act": clip_act,
  }

  @nn.compact
  def __call__(self, x, noise_labels, class_labels):
    cblock = [self.model_channels * x for x in self.channel_mult]
    cnoise = (
      self.model_channels * self.channel_mult_noise
      if self.channel_mult_noise is not None
      else cblock[0]
    )
    cemb = (
      self.model_channels * self.channel_mult_emb
      if self.channel_mult_emb is not None
      else max(cblock)
    )

    if self.out_gain is None:
      out_gain = self.param("out_gain", jax.nn.initializers.zeros, (1,))
    else:
      out_gain = self.out_gain

    # Encoder
    enc = {}
    cout = self.img_channels + 1
    for level, channels in enumerate(cblock):
      res = self.img_resolution >> level
      if level == 0:
        cin = cout
        cout = channels
        enc[f"{res}x{res}_conv"] = MPConv(
          cin, cout, kernel_shape=(3, 3), name=f"enc_{res}x{res}_conv"
        )
      else:
        enc[f"{res}x{res}_down"] = Block(
          cout,
          cout,
          cemb,
          flavor="enc",
          resample_mode="down",
          name=f"enc_{res}x{res}_down",
          **self.block_kwargs,
        )
      for idx in range(self.num_blocks):
        cin = cout
        cout = channels
        enc[f"{res}x{res}_block{idx}"] = Block(
          cin,
          cout,
          cemb,
          flavor="enc",
          attention=(res in self.attn_resolutions),
          name=f"enc_{res}x{res}_block{idx}",
          **self.block_kwargs,
        )

    # Decoder
    dec = {}
    skips = [block.out_channels for block in enc.values()]
    for level, channels in reversed(list(enumerate(cblock))):
      res = self.img_resolution >> level
      if level == len(cblock) - 1:
        dec[f"{res}x{res}_in0"] = Block(
          cout,
          cout,
          cemb,
          flavor="dec",
          attention=True,
          name=f"dec_{res}x{res}_in0",
          **self.block_kwargs,
        )
        dec[f"{res}x{res}_in1"] = Block(
          cout,
          cout,
          cemb,
          flavor="dec",
          name=f"dec_{res}x{res}_in1",
          **self.block_kwargs,
        )
      else:
        dec[f"{res}x{res}_up"] = Block(
          cout,
          cout,
          cemb,
          flavor="dec",
          resample_mode="up",
          name=f"dec_{res}x{res}_up",
          **self.block_kwargs,
        )
      for idx in range(self.num_blocks + 1):
        cin = cout + skips.pop()
        cout = channels
        dec[f"{res}x{res}_block{idx}"] = Block(
          cin,
          cout,
          cemb,
          flavor="dec",
          attention=(res in self.attn_resolutions),
          name=f"dec_{res}x{res}_block{idx}",
          **self.block_kwargs,
        )

    # Embedding
    emb = MPConv(cnoise, cemb, kernel_shape=(), name="emb_noise")(
      MPFourier(cnoise, name="emb_fourier")(noise_labels)
    )
    if self.label_dim != 0:
      emb = mp_sum(
        emb,
        MPConv(self.label_dim, cemb, kernel_shape=(), name="emb_label")(
          class_labels * jnp.sqrt(class_labels.shape[1])
        ),
        t=self.label_balance,
      )
    emb = mp_silu(emb)

    # Encoder
    x = jax.lax.concatenate([x, jnp.ones_like(x[:, :1])], dimension=1)
    skips = []
    for name, block in enc.items():
      x = block(x) if "conv" in name else block(x, emb)
      skips.append(x)

    # Decoder
    for name, block in dec.items():
      if "block" in name:
        x = mp_cat(x, skips.pop(), t=self.concat_balance)
      x = block(x, emb)
    x = MPConv(cout, self.img_channels, kernel_shape=(3, 3), name="out_conv")(
      x, gain=out_gain
    )
    return x


class Precond(nn.Module):
  """Preconditioning and uncertainty estimation."""

  img_resolution: int  # Image resolution.
  img_channels: int  # Image channels.
  label_dim: int  # Class label dimensionality. 0 = unconditional.
  # **precond_kwargs
  use_fp16: bool = True  # Run the model at FP16 precision?
  sigma_data: float = 0.5  # Expected standard deviation of the training data.
  logvar_channels: int = 128  # Intermediate dimensionality for uncertainty estimation.
  return_logvar: bool = False
  # **unet_kwargs  # Keyword arguments for UNet.
  model_channels: int = 192  # Base multiplier for the number of channels.
  channel_mult: tuple = (
    1,
    2,
    3,
    4,
  )  # Per-resolution multipliers for the number of channels.
  channel_mult_noise: Any = None  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
  channel_mult_emb: Any = None  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
  num_blocks: int = 3  # Number of residual blocks per resolution.
  attn_resolutions: tuple = (16, 8)  # List of resolutions with self-attention.
  label_balance: float = (
    0.5  # Balance between noise embedding (0) and class embedding (1).
  )
  concat_balance: float = 0.5  # Balance between skip connections (0) and main path (1).
  out_gain: float = 1.0
  unet_kwargs = {
    "model_channels": model_channels,
    "channel_mult": channel_mult,
    "channel_mult_noise": channel_mult_noise,
    "channel_mult_emb": channel_mult_emb,
    "num_blocks": num_blocks,
    "attn_resolutions": attn_resolutions,
    "label_balance": label_balance,
    "concat_balance": concat_balance,
    "out_gain": out_gain,
  }

  # **block_kwargs  # Keyword arguments for Block
  resample_filter: tuple = (1, 1)  # Resampling filter
  channels_per_head: int = 64  # Number of channels per attention head
  dropout: float = 0.0  # Dropout probability
  res_balance: float = 0.3  # Balance between main branch (0) and residual branch (1)
  attn_balance: float = 0.3  # Balance between main branch (0) and self-attention (1)
  clip_act: int = 256  # Clip output activations. None = do not clip
  out_gain: Any = None
  block_kwargs = {
    "resample_filter": resample_filter,
    "channels_per_head": channels_per_head,
    "dropout": dropout,
    "res_balance": res_balance,
    "attn_balance": attn_balance,
    "clip_act": clip_act,
  }

  @nn.compact
  def __call__(
    self,
    x,
    sigma,
    class_labels=None,
    force_fp32=False,
  ):
    x = jnp.float32(x)
    sigma = jnp.float32(sigma).reshape(-1, 1, 1, 1)
    class_labels = (
      None
      if self.label_dim == 0
      else jnp.zeros((1, self.label_dim), device=x.device)
      if class_labels is None
      else jnp.float32(class_labels).reshape(-1, self.label_dim)
    )
    dtype = jnp.float16 if (self.use_fp16 and not force_fp32) else jnp.float32

    # Preconditioning weights
    c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
    c_out = sigma * self.sigma_data / jnp.sqrt(sigma**2 + self.sigma_data**2)
    c_in = 1 / jnp.sqrt(self.sigma_data**2 + sigma**2)
    c_noise = jnp.log(sigma.flatten()) / 4

    # Run the model
    x_in = jnp.array(c_in * x, dtype=dtype)

    F_x = UNet(
      img_resolution=self.img_resolution,
      img_channels=self.img_channels,
      label_dim=self.label_dim,
      **self.unet_kwargs,
      **self.block_kwargs,
      name="unet",
    )(x_in, c_noise, class_labels)
    D_x = c_skip * x + c_out * jnp.float32(F_x)

    # Estimate uncertainty if requested
    if self.return_logvar:
      logvar = MPConv(self.logvar_channels, 1, kernel_shape=(), name="logvar_linear")(
        MPFourier(self.logvar_channels, name="logvar_fourier")(c_noise)
      ).reshape(-1, 1, 1, 1)
      return D_x, logvar  # u(sigma) in Equation 21
    return D_x
