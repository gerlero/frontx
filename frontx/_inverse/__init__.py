from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .fit import ScaledSolution
from .interpolated import InterpolatedSolution
from .param import Param

__all__ = ["InterpolatedSolution", "Param", "ScaledSolution", "sorptivity"]


def sorptivity(
    o: jax.Array | np.ndarray[Any, Any],
    theta: jax.Array | np.ndarray[Any, Any],
    /,
    *,
    b: float,
    i: float,
) -> jax.Array:
    o = jnp.insert(o, 0, 0)
    theta = jnp.insert(theta, 0, b)

    return jnp.trapezoid(theta - i, o)
