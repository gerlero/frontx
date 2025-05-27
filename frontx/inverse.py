from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from interpax import PchipInterpolator

from ._boltzmann import AbstractSolution, boltzmannmethod


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


T = TypeVar("T", bound=AbstractSolution)


class ScaledSolution(AbstractSolution, Generic[T]):
    original: T
    D0: float | jax.Array

    def __init__(
        self,
        original: T,
        /,
        D0: float | jax.Array,  # noqa: N803
    ) -> None:
        self.original = original
        self.D0 = D0

    @staticmethod
    def with_sorptivity(original: T, S: float | jax.Array, /) -> "ScaledSolution[T]":  # noqa: N803
        return ScaledSolution(
            original,
            D0=(S / original.sorptivity()) ** 2,
        )

    @staticmethod
    def fitting_data(
        original: T,
        o: jax.Array | np.ndarray[Any, Any],
        theta: jax.Array | np.ndarray[Any, Any],
        /,
        sigma: float | jax.Array | np.ndarray[Any, Any] = 1,
    ) -> "ScaledSolution[T]":
        def residuals(
            D0: float | jax.Array,  # noqa: N803,
            args: None = None,  # noqa: ARG001
        ) -> jax.Array:
            scaled = ScaledSolution(original, D0)
            return (scaled(o) - theta) / sigma

        D0 = optx.least_squares(  # noqa: N806
            residuals,
            optx.LevenbergMarquardt(atol=1e-6, rtol=1e-3),
            y0=jnp.array((o[-1] / original.oi) ** 2),
        ).value

        return ScaledSolution(original, D0)

    @boltzmannmethod
    def __call__(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self.original(o / jnp.sqrt(self.D0))

    @boltzmannmethod
    def d_do(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self.original.d_do(o / jnp.sqrt(self.D0)) / jnp.sqrt(self.D0)

    def D(  # noqa: N802
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self.original.D(theta) * self.D0

    @property
    def oi(self) -> float:
        return self.original.oi * jnp.sqrt(self.D0)
