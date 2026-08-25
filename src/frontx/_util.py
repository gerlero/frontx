from collections.abc import Callable
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np


def vmap(
    func: Callable[
        [
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
        ],
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ],
    /,
) -> Callable[
    [float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]],
    float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
]:
    vfunc = jax.vmap(func)

    @wraps(func)
    def vmap_wrapper(
        x: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        if jnp.ndim(x) == 0:
            return func(x)

        return vfunc(x)

    return vmap_wrapper
