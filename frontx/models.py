from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import numpy as np

from ._util import vmap
from .neural import Param


class _MoistureDiffusivityModel(eqx.Module):
    theta_range: eqx.AbstractVar[tuple[float | Param, float | Param]]

    def _Se(  # noqa: N802
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return (theta - self.theta_range[0]) / (
            self.theta_range[1] - self.theta_range[0]
        )

    @abstractmethod
    def __call__(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        raise NotImplementedError


class LETd(_MoistureDiffusivityModel):
    L: float | Param
    E: float | Param
    T: float | Param
    Dwt: float | Param = 1
    theta_range: tuple[float | Param, float | Param] = (0, 1)

    def __call__(
        self, theta: float | jax.Array | np.ndarray[Any, Any]
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = (theta - self.theta_range[0]) / (self.theta_range[1] - self.theta_range[0])  # noqa: N806
        return self.Dwt * Se**self.L / (Se**self.L + self.E * (1 - Se) ** self.T)


class _RichardsModel(_MoistureDiffusivityModel):
    Ks: eqx.AbstractVar[float | Param | None]
    k: eqx.AbstractVar[float | Param | None]
    g: eqx.AbstractVar[float | Param]
    rho: eqx.AbstractVar[float | Param]
    mu: eqx.AbstractVar[float | Param]

    @property
    def _Ks(self) -> float | jax.Array:  # noqa: N802
        if self.Ks is None:
            if self.k is None:
                return 1
            return self.rho * self.g * self.k / self.mu

        if self.k is not None:
            msg = "Cannot set both Ks and k"
            raise ValueError(msg)
        return self.Ks

    def __call__(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self._K(theta) / self._C(theta)

    def _C(  # noqa: N802
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return 1 / vmap(jax.grad(self._h))(theta)

    @abstractmethod
    def _h(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def _kr(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        raise NotImplementedError

    def _K(  # noqa: N802
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return self._Ks * self._kr(theta)


class BrooksAndCorey(_RichardsModel):
    n: float | Param
    l: float | Param = 1  # noqa: E741
    Ks: float | Param | None = None
    k: float | None = None
    g: float | Param = 9.81
    rho: float | Param = 1e3
    mu: float | Param = 1e-3
    alpha: float | Param = 1
    theta_range: tuple[float | Param, float | Param] = (0, 1)

    def _h(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = self._Se(theta)  # noqa: N806
        return -1 / (self.alpha * Se ** (1 / self.n))

    def _kr(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = self._Se(theta)  # noqa: N806
        return Se ** (2 / self.n + self.l + 2)


class VanGenuchten(_RichardsModel):
    n: float | Param | None = None
    m: float | Param | None = None
    l: float | Param = 0.5  # noqa: E741
    Ks: float | Param | None = None
    k: float | Param | None = None
    g: float | Param = 9.81
    rho: float | Param = 1e3
    mu: float | Param = 1e-3
    alpha: float | Param = 1
    theta_range: tuple[float | Param, float | Param] = (0, 1)

    @property
    def _n(self) -> float | jax.Array:
        if self.n is not None:
            return self.n

        if self.m is None:
            msg = "Either n or m must be set"
            raise ValueError(msg)

        return 1 / (1 - self.m)

    @property
    def _m(self) -> float | jax.Array:
        if self.m is not None:
            return self.m

        if self.n is None:
            msg = "Either n or m must be set"
            raise ValueError(msg)

        return 1 - 1 / self.n

    def _h(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = self._Se(theta)  # noqa: N806
        return -((1 / (Se ** (1 / self._m)) - 1) ** (1 / self._n)) / self.alpha

    def _kr(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = self._Se(theta)  # noqa: N806
        return Se**self.l * (1 - (1 - Se ** (1 / self._m)) ** self._m) ** 2


class LETxs(_RichardsModel):
    Lw: float | Param
    Ew: float | Param
    Tw: float | Param
    Ls: float | Param
    Es: float | Param
    Ts: float | Param
    Ks: float | Param | None = None
    k: float | None = None
    g: float | Param = 9.81
    rho: float | Param = 1e3
    mu: float | Param = 1e-3
    alpha: float | Param = 1
    theta_range: tuple[float | Param, float | Param] = (0, 1)

    def _kr(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = self._Se(theta)  # noqa: N806
        return Se**self.Lw / (Se**self.Lw + self.Ew * (1 - Se) ** self.Tw)

    def _h(
        self,
        theta: float | jax.Array | np.ndarray[Any, Any],
        /,
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        Se = self._Se(theta)  # noqa: N806
        return (
            -((1 - Se) ** self.Ls / ((1 - Se) ** self.Ls + self.Es * Se**self.Ts))
            / self.alpha
        )
