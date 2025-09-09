from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from interpax import PchipInterpolator

from frontx._boltzmann import AbstractSolution, boltzmannmethod


class InterpolatedSolution(AbstractSolution):
    oi: float
    _sol: PchipInterpolator
    _do_dtheta: PchipInterpolator
    _Iodtheta: PchipInterpolator
    _c: float

    def __init__(
        self,
        o: jax.Array | np.ndarray[Any, Any],
        theta: jax.Array | np.ndarray[Any, Any],
        /,
        *,
        b: float | None = None,
        i: float | None = None,
    ) -> None:
        o = jnp.asarray(o)
        theta = jnp.asarray(theta)

        self._sol = PchipInterpolator(x=o, y=theta, check=False)

        if b is not None:
            o = jnp.insert(o, 0, 0)
            theta = jnp.insert(theta, 0, b)

        if i is not None:
            o = jnp.append(o, o[-1] + 1)
            theta = jnp.append(theta, i)
        else:
            i = theta[-1]  # ty: ignore[invalid-assignment]

        self.oi: float = o[-1]

        theta, indices = jnp.unique(theta, return_index=True)
        o = o[indices]

        inverse = PchipInterpolator(x=theta, y=o, extrapolate=False, check=False)
        self._do_dtheta = inverse.derivative()
        self._Iodtheta = inverse.antiderivative()
        self._c = self._Iodtheta(i)

    @boltzmannmethod
    def __call__(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self._sol(o)

    def D(  # noqa: N802
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Iodtheta = self._Iodtheta(theta) - self._c  # noqa: N806
        do_dtheta = self._do_dtheta(theta)

        return jnp.squeeze(-(do_dtheta * Iodtheta) / 2)

    def sorptivity(
        self, o: float | jax.Array | np.ndarray[Any, Any] = 0
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Ithetado = self._sol.antiderivative()  # noqa: N806
        return (Ithetado(self.oi) - Ithetado(o)) - self.i * (self.oi - o)
