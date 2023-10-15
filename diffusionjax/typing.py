"""Common types used in diffusionjax."""
from typing import Tuple
import jaxtyping
import typeguard


Shape = Tuple[int, ...]
typed = lambda fn: jaxtyping.jaxtyped(typeguard.typechecked(fn))
