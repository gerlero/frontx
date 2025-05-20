from collections.abc import Callable
from functools import wraps
from typing import Any, overload

import jax
import jax.numpy as jnp
import numpy as np


@overload
def vmap(
    func: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        jax.Array | np.ndarray[Any, Any],
    ],
    /,
) -> Callable[
    [float | jax.Array | np.ndarray[Any, Any]], jax.Array | np.ndarray[Any, Any]
]: ...


@overload
def vmap(
    func: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    /,
) -> Callable[
    [float | jax.Array | np.ndarray[Any, Any]], float | jax.Array | np.ndarray[Any, Any]
]: ...


def vmap(
    func: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    /,
) -> Callable[
    [float | jax.Array | np.ndarray[Any, Any]], float | jax.Array | np.ndarray[Any, Any]
]:
    vfunc = jax.vmap(func)

    @wraps(func)
    def vmap_wrapper(
        x: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        if jnp.ndim(x) == 0:
            return func(x)

        return vfunc(x)

    return vmap_wrapper
