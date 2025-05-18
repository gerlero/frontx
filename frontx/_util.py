from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def vmap(
    func: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    /,
) -> Callable[[float | jax.Array | np.ndarray[Any, Any]], float | jax.Array]:
    @wraps(func)
    def vmap_wrapper(
        x: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        if jnp.ndim(x) == 0:
            return func(x)

        return jax.vmap(func)(x)

    return vmap_wrapper
