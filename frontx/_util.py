from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def vmap(
    func: Callable[
        [float | jax.Array],
        float | jax.Array,
    ],
    /,
) -> Callable[[float | jax.Array | np.ndarray[Any, Any]], float | jax.Array]:
    @wraps(func)
    def vmap_wrapper(
        x: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array:
        if jnp.ndim(x) == 0:
            return func(x)  # type: ignore[arg-type]

        return jax.vmap(func)(x)  # type: ignore[arg-type]

    return vmap_wrapper
