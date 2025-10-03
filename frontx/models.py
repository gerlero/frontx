"""Moisture diffusivity models (public API).

This module exposes parametrized models to compute a moisture
diffusivity-like function ``D(theta)`` used by the PINN solver.

Public classes:
    - LETd: Empirical LET model on effective saturation.
    - BrooksAndCorey: Richards-based model with Brooks & Corey relations.
    - VanGenuchten: Richards-based model with van Genuchten–Mualem relations.
    - LETxs: Richards-based model combining LET-shaped kr and head.

All classes are lightweight `equinox.Module`s and can be called like functions:
``D = model(theta)``. Parameters may be floats or `Param` (trainable) objects.
"""

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import numpy as np

from . import Param
from ._util import vmap


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
    """LET moisture diffusivity on effective saturation.

    Computes

    .. math::

        D(\\theta) = D_{wt} \\; \\frac{Se^{L}}{Se^{L} + E (1-Se)^{T}},\\quad
        Se = \\frac{\\theta-\\theta_r}{\\theta_s-\\theta_r}

    where :math:`\\theta_r, \\theta_s` come from ``theta_range``.
    All parameters can be floats or trainable :class:`Param`.

    Attributes:
        L: Exponent for the wet-side shape.
        E: Balance parameter between wet and dry branches.
        T: Exponent for the dry-side shape.
        Dwt: Scaling factor for the diffusivity (default 1).
        theta_range: Tuple ``(theta_r, theta_s)`` used to compute ``Se``.
    """

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
    """Richards-based model using Brooks & Corey relations.

    The model defines a relative conductivity :math:`k_r(Se)` and capillary
    pressure head :math:`h(Se)` following Brooks & Corey, and computes a
    diffusivity via :math:`D(\\theta) = K(\\theta)/C(\\theta)` with
    :math:`K = K_s k_r` and :math:`C = (\\mathrm{d}h/\\mathrm{d}\\theta)^{-1}`.

    Set either ``Ks`` (saturated conductivity) **or** ``k`` (intrinsic
    permeability) together with fluid properties (``rho``, ``mu``, ``g``).
    If both are provided, a ``ValueError`` is raised.

    Attributes:
        n: Pore-size index (Brooks–Corey exponent).
        l: Mualem connectivity parameter (default 1).
        Ks: Saturated hydraulic conductivity (optional if ``k`` is set).
        k: Intrinsic permeability (mutually exclusive with ``Ks``).
        g: Gravity acceleration (default 9.81).
        rho: Fluid density (default 1e3).
        mu: Dynamic viscosity (default 1e-3).
        alpha: Scaling parameter for pressure head (1/length).
        theta_range: Tuple ``(theta_r, theta_s)`` for effective saturation.
    """

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
    """Richards-based model using van Genuchten–Mualem relations.

    You must set **either** ``n`` or ``m`` (the other is inferred through
    ``m = 1 - 1/n``). The model computes

    .. math::

        h(Se) = -\\frac{1}{\\alpha}\\Big((Se^{-1/m}-1)^{1/n}\\Big),\\qquad
        k_r(Se) = Se^l\\big(1-(1-Se^{1/m})^m\\big)^2

    and returns :math:`D(\\theta)=K(\\theta)/C(\\theta)` as in the base class.

    Attributes:
        n: van Genuchten shape parameter (optional if ``m`` is set).
        m: van Genuchten shape parameter (optional if ``n`` is set).
        l: Mualem connectivity parameter (default 0.5).
        Ks: Saturated hydraulic conductivity (optional if ``k`` is set).
        k: Intrinsic permeability (mutually exclusive with ``Ks``).
        g: Gravity acceleration (default 9.81).
        rho: Fluid density (default 1e3).
        mu: Dynamic viscosity (default 1e-3).
        alpha: Scaling parameter for pressure head (1/length).
        theta_range: Tuple ``(theta_r, theta_s)`` for effective saturation.

    Raises:
        ValueError: If neither ``n`` nor ``m`` is provided.
    """

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
    """Richards-based LET model with separate wet/dry shapes.

    Uses LET-shaped functions both for relative conductivity and pressure head:

    .. math::

        k_r(Se) = \\frac{Se^{L_w}}{Se^{L_w} + E_w (1-Se)^{T_w}},\\qquad
        h(Se) = -\\frac{(1-Se)^{L_s}}{(1-Se)^{L_s} + E_s Se^{T_s}}\\,\\frac{1}{\\alpha}

    and returns :math:`D(\\theta)=K(\\theta)/C(\\theta)`.

    Attributes:
        Lw: Wet-side exponent in ``k_r``.
        Ew: Balance parameter in ``k_r``.
        Tw: Dry-side exponent in ``k_r``.
        Ls: Dry-side exponent in ``h``.
        Es: Balance parameter in ``h``.
        Ts: Wet-side exponent in ``h``.
        Ks: Saturated hydraulic conductivity (optional if ``k`` is set).
        k: Intrinsic permeability (mutually exclusive with ``Ks``).
        g: Gravity acceleration (default 9.81).
        rho: Fluid density (default 1e3).
        mu: Dynamic viscosity (default 1e-3).
        alpha: Scaling parameter for pressure head (1/length).
        theta_range: Tuple ``(theta_r, theta_s)`` for effective saturation.
    """

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
