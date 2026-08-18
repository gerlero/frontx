from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Protocol, TypeVar, overload

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._util import vmap


def ode(
    D: Callable[
        [
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
        ],
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ],
) -> diffrax.ODETerm[jax.Array]:
    @diffrax.ODETerm[jax.Array]  # ty: ignore[invalid-argument-type]
    def term(
        o: jax.Array,
        y: jax.Array,
        args: None,
    ) -> jax.Array:
        theta, dtheta_do = y

        D_, dD_dtheta = jax.value_and_grad(D)(theta)

        d2theta_do2 = -((o / 2 + dD_dtheta * dtheta_do) / D_) * dtheta_do

        return jnp.array([dtheta_do, d2theta_do2])

    return term


_Self = TypeVar("_Self")


class _BoltzmannTransformed(Protocol):
    @overload
    def __call__(
        self,
        r: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        t: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array: ...

    @overload
    def __call__(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array: ...


def boltzmannmethod(
    meth: Callable[
        [
            _Self,
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        ],
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ],
    /,
) -> _BoltzmannTransformed:
    @overload
    def boltzmann_wrapper(
        self: _Self,
        r: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        t: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> (
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
    ): ...

    @overload
    def boltzmann_wrapper(
        self: _Self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> (
        float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
    ): ...

    @wraps(meth)
    def boltzmann_wrapper(
        self: _Self,
        *args: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        **kwargs: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        match args, kwargs:
            case (o,), {} if not kwargs:
                pass
            case (r, t), {} if not kwargs:
                o = r / jnp.sqrt(t)
            case (), {"o": o} if len(kwargs) == 1:
                pass
            case (), {"r": r, "t": t} if len(kwargs) == 2:
                o = r / jnp.sqrt(t)
            case _:
                msg = f"{getattr(meth, '__name__', 'method')} takes either (r, t) or (o,) as arguments"
                raise TypeError(msg)

        return meth(self, o)

    return boltzmann_wrapper  # ty: ignore[invalid-return-type]


class AbstractSolution(eqx.Module):
    D: eqx.AbstractVar[
        Callable[
            [
                float
                | jax.Array
                | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]
            ],
            float
            | jax.Array
            | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        ]
    ]
    oi: eqx.AbstractVar[float]

    @property
    def b(self) -> jax.Array:
        return self(0.0)

    @property
    def d_dob(self) -> jax.Array:
        return self.d_do(0.0)

    @property
    def i(self) -> jax.Array:
        return self(self.oi)

    @abstractmethod
    @boltzmannmethod
    def __call__(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        raise NotImplementedError

    @boltzmannmethod
    def d_do(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return vmap(jax.grad(self))(o)

    def d_dr(
        self,
        r: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        t: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array:
        return self.d_do(r, t) / jnp.sqrt(t)

    def d_dt(
        self,
        r: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        t: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array:
        return -r / (jnp.sqrt(t) * 2 * t) * self.d_do(r, t)

    def flux(
        self,
        r: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        t: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
    ) -> jax.Array:
        return -self.D(self(r, t)) * self.d_dr(r, t)  # ty: ignore[invalid-return-type]

    def sorptivity(
        self,
        o: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]] = 0.0,
    ) -> jax.Array:
        return -2 * self.D(self(o)) * self.d_do(o)  # ty: ignore[invalid-return-type]
