from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import RESULTS


class Solution(eqx.Module):
    D: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]
    r1: float
    t1: float
    _sol: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]

    def __call__(
        self,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        theta = self._sol.evaluate(t)
        return jnp.interp(r, jnp.linspace(0, self.r1, theta.size), theta)

    @property
    def result(self) -> RESULTS:
        return self._sol.result


@eqx.filter_jit
def solve(  # noqa: PLR0913
    D: Callable[  # noqa: N803
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    r1: float,
    t1: float,
    *,
    i: jax.Array | np.ndarray[Any, Any],
    b: float | None = None,
    throw: bool = True,
) -> Solution:
    i = jnp.asarray(i)
    dr = r1 / (i.size - 1)
    dr2 = dr**2

    @diffrax.ODETerm[jax.Array]
    def term(
        _: float,
        theta: jax.Array,
        args: None,  # noqa: ARG001
    ) -> jax.Array:
        D_ = jnp.asarray(D(theta))  # noqa: N806
        if D_.ndim == 0:
            D_ = jnp.repeat(D_, theta.size)  # noqa: N806
        Df = (D_[1:] + D_[:-1]) / 2  # noqa: N806

        return jnp.concatenate(
            [
                jnp.array([Df[0] / dr2 * (theta[1] - theta[0]) if b is None else 0.0]),
                Df[:-1] / dr2 * theta[:-2]
                - (Df[1:] + Df[:-1]) / dr2 * theta[1:-1]
                + Df[1:] / dr2 * theta[2:],
                jnp.array([Df[-1] / dr2 * (theta[-2] - theta[-1])]),
            ]
        )

    sol = diffrax.diffeqsolve(
        term,
        solver=diffrax.Kvaerno5(),
        t0=0,
        t1=t1,
        dt0=None,
        y0=i if b is None else i.at[0].set(b),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(t0=True, t1=True, dense=True),
        throw=throw,
    )

    return Solution(
        D=D,
        r1=r1,
        t1=t1,
        _sol=sol,
    )
