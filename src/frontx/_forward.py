from collections.abc import Callable
from typing import cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from ._boltzmann import AbstractSolution, boltzmannmethod, ode
from ._util import vmap

RESULTS = diffrax.RESULTS

_Diffusivity = Callable[
    [
        float
        | jax.Array
        | np.ndarray[tuple[int, ...], np.dtype[np.floating | np.integer]]
    ],
    float | jax.Array | np.ndarray[tuple[int, ...], np.dtype[np.floating | np.integer]],
]
_DiffusivityInput = (
    _Diffusivity
    | Callable[
        [
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
        ],
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ]
)


class Solution(AbstractSolution):
    _sol: diffrax.Solution
    result: RESULTS
    D: _Diffusivity

    @boltzmannmethod
    def __call__(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return vmap(self._sol.evaluate)(jnp.clip(o, 0, self.oi))[..., 0]  # ty: ignore[not-subscriptable]

    @boltzmannmethod
    def d_do(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return vmap(self._sol.evaluate)(jnp.clip(o, 0, self.oi))[..., 1]  # ty: ignore[not-subscriptable]

    @property
    def oi(self) -> jax.Array:
        assert self._sol.ts is not None
        return self._sol.ts[-1]

    @property
    def i(self) -> jax.Array:
        assert self._sol.ys is not None
        return self._sol.ys[-1, 0]

    @property
    def b(self) -> jax.Array:
        assert self._sol.ys is not None
        return self._sol.ys[0, 0]

    @property
    def d_dob(self) -> jax.Array:
        assert self._sol.ys is not None
        return self._sol.ys[0, 1]


@eqx.filter_jit
def solve(
    D: _DiffusivityInput,
    *,
    b: float | jax.Array,
    i: float | jax.Array,
    itol: float = 1e-3,
    max_steps: int = 100,
    throw: bool = True,
) -> Solution:
    term = ode(cast(_Diffusivity, D))
    direction = jnp.sign(i - b)

    @diffrax.Event
    def event(t: float, y: jax.Array, args: object, **kwargs: object) -> jax.Array:
        return (direction * y[1] <= 0) | (direction * y[0] > direction * (i - itol))

    def shoot(
        d_dob: float | jax.Array,
        args: None,
    ) -> tuple[jax.Array, diffrax.Solution]:
        sol = diffrax.diffeqsolve(
            term,
            solver=diffrax.Kvaerno5(),
            t0=0,
            t1=jnp.inf,
            dt0=None,
            y0=jnp.array([b, d_dob]),
            event=event,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(t0=True, t1=True, dense=True),
            throw=False,
        )
        assert sol.ys is not None
        residual = jax.lax.select(
            sol.result == diffrax.RESULTS.event_occurred,
            sol.ys[-1, 0] - i,
            direction * jnp.inf,
        )
        return residual, sol

    root: optx.Solution = optx.root_find(
        shoot,
        solver=optx.Bisection(rtol=jnp.inf, atol=itol, expand_if_necessary=True),
        y0=0,
        max_steps=max_steps,
        has_aux=True,
        options={"lower": 0, "upper": (i - b) / (2 * jnp.sqrt(D(b)))},
        throw=throw,
    )

    return Solution(
        _sol=root.aux,
        result=RESULTS.where(
            root.result == optx.RESULTS.successful,
            RESULTS.successful,
            RESULTS.max_steps_reached,
        ),
        D=cast(_Diffusivity, D),
    )  # ty: ignore[missing-argument]
