from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from frontx._boltzmann import AbstractSolution, boltzmannmethod

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

    @eqx.filter_jit
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
