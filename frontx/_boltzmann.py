from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol, TypeVar, overload

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._util import vmap


def ode(
    D: Callable[  # noqa: N803
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
) -> diffrax.ODETerm[jax.Array]:
    @diffrax.ODETerm[jax.Array]
    def term(
        o: bool | float | jax.Array | np.ndarray[Any, Any],
        y: jax.Array,
        args: None,  # noqa: ARG001
    ) -> jax.Array:
        theta, dtheta_do = y

        D_, dD_dtheta = jax.value_and_grad(D)(theta)  # noqa: N806

        d2theta_do2 = -((o / 2 + dD_dtheta * dtheta_do) / D_) * dtheta_do

        return jnp.array([dtheta_do, d2theta_do2])

    return term


T = TypeVar("T")


class _BoltzmannTransformed(Protocol):
    @overload
    def __call__(
        self,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]: ...

    @overload
    def __call__(
        self, o: float | jax.Array | np.ndarray[Any, Any]
    ) -> float | jax.Array | np.ndarray[Any, Any]: ...


def boltzmannmethod(
    meth: Callable[
        [T, float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
) -> _BoltzmannTransformed:
    @overload
    def boltzmann_wrapper(
        self: T,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]: ...

    @overload
    def boltzmann_wrapper(
        self: T, o: float | jax.Array | np.ndarray[Any, Any]
    ) -> float | jax.Array | np.ndarray[Any, Any]: ...

    @wraps(meth)
    def boltzmann_wrapper(
        self: T,
        *args: float | jax.Array | np.ndarray[Any, Any],
        **kwargs: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        if len(args) == 1 and not kwargs:
            o = args[0]
        elif len(args) == 2 and not kwargs:  # noqa: PLR2004
            r, t = args
            o = r / jnp.sqrt(t)
        elif not args and "o" in kwargs:
            o = kwargs["o"]
        elif not args and "r" in kwargs and "t" in kwargs:
            r = kwargs["r"]
            t = kwargs["t"]
            o = r / jnp.sqrt(t)
        else:
            meth_name = getattr(meth, "__name__", "method")
            msg = f"{meth_name} takes (r, t) or (o,) as arguments"
            raise TypeError(msg)

        return meth(self, o)

    return boltzmann_wrapper


class AbstractSolution(eqx.Module):
    oi: eqx.AbstractVar[float]

    @property
    def b(self) -> float | jax.Array | np.ndarray[Any, Any]:
        return self(0.0)

    @property
    def d_dob(self) -> float | jax.Array | np.ndarray[Any, Any]:
        return self.d_do(0.0)

    @property
    def i(self) -> float | jax.Array | np.ndarray[Any, Any]:
        return self(self.oi)

    @abstractmethod
    @boltzmannmethod
    def __call__(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        raise NotImplementedError

    @boltzmannmethod
    def d_do(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return vmap(jax.grad(self))(o)

    def d_dr(
        self,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self.d_do(r, t) / jnp.sqrt(t)

    def d_dt(
        self,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return -r / (jnp.sqrt(t) * 2 * t) * self.d_do(r, t)

    def D(  # noqa: N802
        self,
        _: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        raise NotImplementedError

    def flux(
        self,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return -self.D(self(r, t)) * self.d_dr(r, t)

    def sorptivity(
        self, o: float | jax.Array | np.ndarray[Any, Any] = 0.0
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return -2 * self.D(self(o)) * self.d_do(o)

    def residual(
        self, o: float | jax.Array | np.ndarray[Any, Any]
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        lhs = -o / 2 * self.d_do(o)
        rhs = vmap(jax.grad(lambda o: (self.D(self(o)) * self.d_do(o))))(o)
        return lhs - rhs
