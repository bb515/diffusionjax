"""Common types used in diffusionjax."""
from typing import Tuple
import jaxtyping
import typeguard


Shape = Tuple[int, ...]
def typed(fn): return jaxtyping.jaxtyped(typeguard.typechecked(fn))
