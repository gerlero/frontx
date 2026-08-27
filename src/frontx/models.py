"""Moisture diffusivity models."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import Param
from ._util import vmap


class _MoistureDiffusivityModel(eqx.Module):
    theta_range: eqx.AbstractVar[tuple[float | Param, float | Param]]

    def _Se(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return (theta - self.theta_range[0]) / (
            self.theta_range[1] - self.theta_range[0]
        )

    @abstractmethod
    def __call__(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
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

    L: float | Param  # ty: ignore[dataclass-field-order]
    E: float | Param  # ty: ignore[dataclass-field-order]
    T: float | Param  # ty: ignore[dataclass-field-order]
    Dwt: float | Param = 1
    theta_range: tuple[float | Param, float | Param] = (0, 1)

    def __call__(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = (theta - self.theta_range[0]) / (self.theta_range[1] - self.theta_range[0])
        return self.Dwt * Se**self.L / (Se**self.L + self.E * (1 - Se) ** self.T)


class _RichardsModel(_MoistureDiffusivityModel):
    Ks: eqx.AbstractVar[float | jax.Array | Param | None]
    k: eqx.AbstractVar[float | jax.Array | Param | None]
    g: eqx.AbstractVar[float | jax.Array | Param]
    rho: eqx.AbstractVar[float | jax.Array | Param]
    mu: eqx.AbstractVar[float | jax.Array | Param]

    @property
    def _Ks(self) -> float | jax.Array | Param:
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
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return self._K(theta) / self._C(theta)  # ty: ignore[invalid-return-type]

    def _C(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        return 1 / vmap(jax.grad(self._h))(theta)

    @abstractmethod
    def _h(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        raise NotImplementedError

    @abstractmethod
    def _kr(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        raise NotImplementedError

    def _K(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
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
        n: Pore-size index (Brooks and Corey exponent).
        l: Mualem connectivity parameter (default 1).
        Ks: Saturated hydraulic conductivity (optional if ``k`` is set).
        k: Intrinsic permeability (mutually exclusive with ``Ks``).
        g: Gravity acceleration (default 9.81).
        rho: Fluid density (default 1e3).
        mu: Dynamic viscosity (default 1e-3).
        alpha: Scaling parameter for pressure head (1/length).
        theta_range: Tuple ``(theta_r, theta_s)`` for effective saturation.
    """

    n: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    l: float | jax.Array | Param = 1
    Ks: float | jax.Array | Param | None = None
    k: float | jax.Array | None = None
    g: float | jax.Array | Param = 9.81
    rho: float | jax.Array | Param = 1e3
    mu: float | jax.Array | Param = 1e-3
    alpha: float | jax.Array | Param = 1
    theta_range: tuple[float | jax.Array | Param, float | jax.Array | Param] = (0, 1)

    def _h(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        return -1 / (self.alpha * Se ** (1 / self.n))

    def _kr(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        return Se ** (2 / self.n + self.l + 2)


class VanGenuchten(_RichardsModel):
    """Richards-based model using van Genuchten-Mualem relations.

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

    n: float | jax.Array | Param | None = None
    m: float | jax.Array | Param | None = None
    l: float | jax.Array | Param = 0.5
    Ks: float | jax.Array | Param | None = None
    k: float | jax.Array | Param | None = None
    g: float | jax.Array | Param = 9.81
    rho: float | jax.Array | Param = 1e3
    mu: float | jax.Array | Param = 1e-3
    alpha: float | jax.Array | Param = 1
    theta_range: tuple[float | jax.Array | Param, float | jax.Array | Param] = (0, 1)

    @property
    def _n(self) -> float | jax.Array | Param:
        if self.n is not None:
            return self.n

        if self.m is None:
            msg = "Either n or m must be set"
            raise ValueError(msg)

        return 1 / (1 - self.m)

    @property
    def _m(self) -> float | jax.Array | Param:
        if self.m is not None:
            return self.m

        if self.n is None:
            msg = "Either n or m must be set"
            raise ValueError(msg)

        return 1 - 1 / self.n

    def _h(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        return -((1 / (Se ** (1 / self._m)) - 1) ** (1 / self._n)) / self.alpha

    def _kr(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        return Se**self.l * jnp.expm1(self._m * jnp.log1p(-(Se ** (1 / self._m)))) ** 2

    def __call__(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        u = self._m * jnp.log1p(-(Se ** (1 / self._m)))

        return (
            (1 - self._m)
            * self._Ks
            / (self.alpha * self._m * (self.theta_range[1] - self.theta_range[0]))
            * Se**self.l
            * Se ** (-1 / self._m)
            * jnp.expm1(u) ** 2
            * jnp.exp(-u)
        )


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

    Lw: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    Ew: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    Tw: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    Ls: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    Es: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    Ts: float | jax.Array | Param  # ty: ignore[dataclass-field-order]
    Ks: float | jax.Array | Param | None = None
    k: float | jax.Array | Param | None = None
    g: float | jax.Array | Param = 9.81
    rho: float | jax.Array | Param = 1e3
    mu: float | jax.Array | Param = 1e-3
    alpha: float | jax.Array | Param = 1
    theta_range: tuple[float | jax.Array | Param, float | jax.Array | Param] = (0, 1)

    def _kr(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        return Se**self.Lw / (Se**self.Lw + self.Ew * (1 - Se) ** self.Tw)  # ty: ignore[invalid-return-type]

    def _h(
        self,
        theta: float
        | jax.Array
        | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]],
        /,
    ) -> float | jax.Array | np.ndarray[tuple[int], np.dtype[np.floating | np.integer]]:
        Se = self._Se(theta)
        return (
            -((1 - Se) ** self.Ls / ((1 - Se) ** self.Ls + self.Es * Se**self.Ts))
            / self.alpha
        )
