import pytest
from diffusionjax.utils import batch_mul
import jax.numpy as jnp


def test_batch_mul():
    """Placeholder test for `:meth:batch_mul` to test CI"""
    a = jnp.ones((2,))
    bs = [jnp.zeros((2,)), jnp.ones((2,)), jnp.ones((2,)) * jnp.pi]
    c_expecteds = [jnp.zeros((2,)), jnp.ones((2,)), jnp.ones((2,)) * jnp.pi]
    for i, b in enumerate(bs):
        c = batch_mul(a, b)
        assert jnp.allclose(c, c_expecteds[i])
