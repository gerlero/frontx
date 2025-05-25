from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from interpax import PchipInterpolator

from ._boltzmann import AbstractSolution, boltzmannmethod, ode
from ._util import vmap

__version__ = "0.1.0"


RESULTS = diffrax.RESULTS


class Solution(AbstractSolution):
    _sol: diffrax.Solution
    result: RESULTS
    D: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]

    @boltzmannmethod
    def __call__(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return vmap(self._sol.evaluate)(jnp.clip(o, 0, self.oi))[..., 0]

    @boltzmannmethod
    def d_do(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return vmap(self._sol.evaluate)(jnp.clip(o, 0, self.oi))[..., 1]

    @property
    def oi(self) -> float:
        assert self._sol.ts is not None
        return self._sol.ts[-1]

    @property
    def i(self) -> float:
        assert self._sol.ys is not None
        return self._sol.ys[-1, 0]

    @property
    def b(self) -> float:
        assert self._sol.ys is not None
        return self._sol.ys[0, 0]

    @property
    def d_dob(self) -> float:
        assert self._sol.ys is not None
        return self._sol.ys[0, 1]


@eqx.filter_jit
def solve(  # noqa: PLR0913
    D: Callable[  # noqa: N803
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    *,
    b: float,
    i: float,
    itol: float = 1e-3,
    maxiter: int = 100,
    throw: bool = True,
) -> Solution:
    term = ode(D)
    direction = jnp.sign(i - b)

    @diffrax.Event
    def event(t: float, y: jax.Array, args: object, **kwargs: object) -> jax.Array:  # noqa: ARG001
        return (direction * y[1] <= 0) | (direction * y[0] > direction * (i - itol))

    def shoot(
        d_dob: float | jax.Array,
        args: None,  # noqa: ARG001
    ) -> tuple[jax.Array | diffrax.Solution]:
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
        return residual, sol  # ty: ignore[invalid-return-type]

    root: optx.Solution = optx.root_find(
        shoot,
        solver=optx.Bisection(
            rtol=jnp.inf, atol=itol, expand_if_necessary=True
        ),  # ty: ignore[missing-argument]
        y0=0,
        max_steps=maxiter,
        has_aux=True,
        options={"lower": 0, "upper": (i - b) / (2 * jnp.sqrt(D(b)))},
        throw=throw,
    )

    return Solution(
        root.aux,
        RESULTS.where(
            root.result == optx.RESULTS.successful,
            RESULTS.successful,
            RESULTS.max_steps_reached,
        ),
        D,  # ty: ignore[invalid-argument-type]
    )  # ty: ignore[missing-argument]


class InterpolatedSolution(AbstractSolution):
    oi: float
    _sol: PchipInterpolator
    _do_dtheta: PchipInterpolator
    _Iodtheta: PchipInterpolator
    _c: float
    result: RESULTS

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
        self.result = RESULTS.successful

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
