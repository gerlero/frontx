"""Tests for the finite-difference solver (`frontx.finite`).

This suite validates two core properties:

1) Consistency vs. the reference solver (`frontx.solve`) for LET-family models.
2) Mass conservation: the integrated uptake matches the sorptivity over time.
"""

from collections.abc import Callable
from typing import Any

import frontx
import frontx.finite
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from frontx.models import LETd, LETxs

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@pytest.mark.parametrize(
    "D",
    [
        LETd(L=0.004569, E=12930, T=1.505, Dwt=4.660e-4, theta_range=(0.019852, 0.7)),
        LETxs(
            Lw=1.651,
            Ew=230.5,
            Tw=0.9115,
            Ls=0.517,
            Es=493.6,
            Ts=0.3806,
            Ks=8.900e-3,
            theta_range=(0.01176, 0.7),
        ),
    ],
)
def test_validity_let(
    D: Callable[  # noqa: N803
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
) -> None:
    """Finite solver reproduces the reference solution for LET-type models.

    The test compares the finite-difference solution
    (:func:`frontx.finite.solve`) against the reference solver
    (:func:`frontx.solve`) at a fixed time, using two representative
    diffusivities (LETd, LETxs). Agreement is checked within an absolute
    tolerance at all spatial points.

    Asserts:
        The maximum absolute difference between both solutions is below 2e-2.
    """
    b = 0.7 - 1e-7
    i = 0.025

    ref = frontx.solve(D, b=b, i=i)

    r = jnp.linspace(0, 0.0025, 500)
    t = 1.0

    sol = frontx.finite.solve(D, r[-1], t, b=b, i=jnp.repeat(i, r.size))

    assert sol(r, t) == pytest.approx(ref(r, t), abs=2e-2)


def test_mass_conservation() -> None:
    """Integrated uptake equals sorptivity over time (mass conservation).

    The reference solution provides the sorptivity (total uptake). We then
    evolve the finite solver from the reference profile at ``t=1`` and verify
    that the spatial integral of ``theta(r, t) - i`` stays equal to the total
    sorptivity for all times in a grid.

    Asserts:
        For each sampled time, the trapezoidal integral over ``r`` matches the
        reference sorptivity within 1e-4 absolute tolerance.
    """
    D = LETd(L=0.004569, E=12930, T=1.505, Dwt=4.660e-4, theta_range=(0.019852, 0.7))  # noqa: N806
    b = 0.7 - 1e-7
    i = 0.025

    sol = frontx.solve(D, b=b, i=i)

    r = jnp.linspace(0, 0.0025, 500)
    t = jnp.linspace(0, 1_000, 500)

    total = sol.sorptivity()
    sol = frontx.finite.solve(D, r[-1], t[-1], i=sol(r, 1.0))

    for t_ in t:
        assert jnp.trapezoid(sol(r, t_) - i, r) == pytest.approx(total, abs=1e-4)
