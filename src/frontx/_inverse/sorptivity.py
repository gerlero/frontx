import jax
import jax.numpy as jnp
import numpy as np


def sorptivity(
    o: jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    theta: jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    /,
    *,
    b: float | jax.Array,
    i: float | jax.Array,
) -> jax.Array:
    o = jnp.insert(o, 0, 0)
    theta = jnp.insert(theta, 0, b)

    return jnp.trapezoid(theta - i, o)
