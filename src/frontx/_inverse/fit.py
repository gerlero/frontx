from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from frontx import RESULTS, Solution, solve
from frontx._boltzmann import AbstractSolution, boltzmannmethod

from .param import de_fit
from .sorptivity import sorptivity


class ScaledSolution(AbstractSolution):
    original: AbstractSolution
    D0: float | jax.Array
    result: RESULTS

    def __init__(
        self,
        original: AbstractSolution,
        /,
        D0: float | jax.Array,
        *,
        _result: RESULTS = RESULTS.successful,
    ) -> None:
        self.original = original
        self.D0 = D0
        self.result = _result

    @staticmethod
    def with_sorptivity(
        original: AbstractSolution,
        S: float | jax.Array,
        /,
    ) -> "ScaledSolution":
        return ScaledSolution(
            original,
            D0=(S / original.sorptivity()) ** 2,
        )

    @eqx.filter_jit
    @staticmethod
    def fitting_data(
        original: AbstractSolution,
        o: jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        theta: jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
        sigma: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] = 1,
        *,
        throw: bool = True,
    ) -> "ScaledSolution":
        def residuals(
            D0: float | jax.Array,
            args: None = None,
        ) -> jax.Array:
            scaled = ScaledSolution(original, D0)
            return (scaled(o) - theta) / sigma

        opt = optx.least_squares(
            residuals,
            optx.LevenbergMarquardt(atol=1e-6, rtol=1e-3),
            y0=jnp.array((o[-1] / original.oi) ** 2),
            throw=throw,
        )

        result = RESULTS.where(
            opt.result == optx.RESULTS.successful,
            RESULTS.successful,
            RESULTS.where(
                opt.result == optx.RESULTS.max_steps_reached,
                RESULTS.max_steps_reached,
                RESULTS.internal_error,
            ),
        )

        D0 = opt.value

        return ScaledSolution(original, D0, _result=result)

    @boltzmannmethod
    def __call__(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array:
        return self.original(o / jnp.sqrt(self.D0))

    @boltzmannmethod
    def d_do(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array:
        return self.original.d_do(o / jnp.sqrt(self.D0)) / jnp.sqrt(self.D0)

    def D(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return self.original.D(theta) * self.D0

    @property
    def oi(self) -> jax.Array:
        return self.original.oi * jnp.sqrt(self.D0)


def fit(
    D: Callable[
        [
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
        ],
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ],
    o: jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    theta: jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    /,
    sigma: float
    | jax.Array
    | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] = 1,
    *,
    i: float | jax.Array,
    b: float | jax.Array,
    fit_D0: Literal["data", "sorptivity"] | None = "data",
    max_steps: int = 15,
) -> ScaledSolution | Solution:
    if fit_D0 == "sorptivity":
        S = sorptivity(o, theta, b=b, i=i)

    def candidate(
        D: Callable[
            [
                float
                | jax.Array
                | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
            ],
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        ],
    ) -> ScaledSolution | Solution:
        sol = solve(D, i=i, b=b, throw=False)
        match fit_D0:
            case "data":
                return ScaledSolution.fitting_data(
                    sol, o, theta, sigma=sigma, throw=False
                )
            case "sorptivity":
                return ScaledSolution.with_sorptivity(sol, S)
            case None:
                return sol

    def cost(sol: ScaledSolution | Solution) -> float:
        result = sol.original.result if isinstance(sol, ScaledSolution) else sol.result  # ty: ignore[unresolved-attribute]
        return jax.lax.cond(
            result == RESULTS.successful,
            lambda: jnp.mean(((sol(o) - theta) / sigma) ** 2),
            lambda: jnp.inf,
        )

    return de_fit(candidate, cost, initial=D, max_steps=max_steps)
