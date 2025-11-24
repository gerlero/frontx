"""Finite-difference diffusion solver and data fitting.

This module provides a small public API to (a) integrate a radial 1D diffusion
model with a state-dependent diffusivity ``D(theta)``, and (b) fit that model
to spatio–temporal data. Internally, integration is performed with Diffrax and
a centered finite-difference stencil.

Public API:
- :class:`Solution`: wrapper for evaluating the simulated field and metadata.
- :func:`solve`: integrate the PDE given initial/boundary conditions.
- :func:`fit`: estimate model/parameters by minimizing data misfit.
"""

from collections.abc import Callable
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from frontx._inverse.param import de_fit

from . import RESULTS

# Export only the public API
__all__ = ["Solution", "solve", "fit"]


class Solution(eqx.Module):
    """Simulation result wrapper for the finite-difference model.

    A :class:`Solution` evaluates the predicted field :math:`\\theta(r, t)`
    on-demand via dense output from the ODE integrator and linear interpolation
    across the spatial grid.

    Attributes:
        D: Diffusivity callable used in the simulation (accepts ``theta`` and
            returns ``D(theta)`` with broadcastable shape).
        r1: Domain radius (the grid spans ``r in [0, r1]``).
    """

    D: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]
    r1: float
    _sol: diffrax.Solution

    def __call__(
        self,
        r: float | jax.Array | np.ndarray[Any, Any],
        t: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        """Evaluate the simulated field at coordinates ``(r, t)``.

        Dense time output is queried from the solver and linearly interpolated
        over the spatial grid ``[0, r1]``.

        Args:
            r: Radial position(s) where the field is desired (scalar or array).
            t: Time(s) at which to evaluate (scalar or array). Supports
                broadcasting with ``r``.

        Returns:
            Values of ``theta(r, t)`` with a shape broadcastable from ``r`` and
            ``t``.
        """
        theta = self._sol.evaluate(t)
        return jnp.interp(r, jnp.linspace(0, self.r1, theta.size), theta)

    @property

    def t1(self) -> float | jax.Array | np.ndarray[Any, Any]:
        """Final integration time."""
        return self._sol.t1

    @property
    def result(self) -> RESULTS:
        """Diffrax integration status (see :data:`frontx.RESULTS`)."""
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
    """Integrate the finite-difference diffusion model.

    Solves a radial 1D diffusion-like PDE on ``r ∈ [0, r1]`` up to time ``t1``
    with state-dependent diffusivity ``D(theta)``. The initial condition is
    ``theta(r, 0) = i[r]`` (grid-aligned). At the inner boundary,
    ``r = 0``, a Neumann-like symmetry condition is applied by the stencil.
    At the outer boundary, ``r = r1``, a Neumann condition is used; optionally
    if ``b`` is provided, the first cell is clamped to ``b`` in ``y0`` to mimic
    a Dirichlet-type condition on the inner node.

    Args:
        D: Callable returning ``D(theta)`` (scalar or array) for the current
            state ``theta``.
        r1: Domain radius (maximum ``r``).
        t1: Final time to integrate to.
        i: Initial condition values on the spatial grid (1D array). Its length
            defines the number of spatial nodes.
        b: Optional value to clamp the first grid node at initialization.
        throw: If ``True``, Diffrax raises on integration failures.

    Returns:
        A :class:`Solution` object that can be called as ``sol(r, t)`` and
        exposes ``sol.t1`` and ``sol.result``.

    Notes:
        - Spatial discretization uses a centered three-point stencil with a
            harmonic-like averaging of ``D`` at faces.
        - Time integration uses ``Kvaerno5`` with adaptive step control.
    """
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
        _sol=sol,
    )


def fit(  # noqa: PLR0913
    D: Callable[  # noqa: N803
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    r1: float,
    t1: float,
    r: jax.Array | np.ndarray[Any, Any],
    t: jax.Array | np.ndarray[Any, Any],
    theta: jax.Array | np.ndarray[Any, Any],
    /,
    sigma: float | jax.Array | np.ndarray[Any, Any] = 1,
    *,
    i: jax.Array | np.ndarray[Any, Any],
    b: float | None = None,
    max_steps: int = 15,
) -> Solution:
    """Fit the finite-difference model to space–time observations.

    This routine searches over candidate models/parameters (through
    :func:`frontx._inverse.param.de_fit`) to minimize the mean squared error
    between the simulated field and observations ``theta(r, t)`` with optional
    per-sample weighting ``sigma``.

    Args:
        D: Initial diffusivity callable or parameterized model to start from.
        r1: Domain radius for the simulation.
        t1: Final time for the simulation.
        r: Radial coordinates of observations, shape ``(Nr,)``.
        t: Time coordinates of observations, shape ``(Nt,)``.
        theta: Observed values with shape broadcastable to ``(Nr, Nt)`` or
            exactly ``(Nr, Nt)``.
        sigma: Noise/weight (scalar or array broadcastable to ``theta``),
            used in the weighted MSE.
        i: Initial condition on the grid (length defines spatial resolution).
        b: Optional clamped value for the first grid node at initialization.
        max_steps: Maximum number of differential-evolution iterations.

    Returns:
        A :class:`Solution` corresponding to the best candidate found.

    Notes:
        - Candidates are generated/updated by ``de_fit``; this function defines
            the simulation and cost evaluation only.
        - If an integration attempt is unsuccessful, its cost is ``+inf``.
    """
    def candidate(
        D: Callable[  # noqa: N803
            [float | jax.Array | np.ndarray[Any, Any]],
            float | jax.Array | np.ndarray[Any, Any],
        ],
    ) -> Solution:
        return solve(D, r1, t1, i=i, b=b)

    def cost(sol: Solution) -> float:
        return jax.lax.cond(
            sol.result == RESULTS.successful,
            lambda: jnp.mean(((sol(r, t[:, jnp.newaxis]) - theta) / sigma) ** 2),
            lambda: jnp.inf,
        )

    return de_fit(candidate, cost, initial=D, max_steps=max_steps)
