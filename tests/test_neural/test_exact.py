"""Exact solution test for the neural PINN (`frontx.neural.fit`).

This verifies the classic Philip (1960) case:
    D(theta) = (1 - log(theta)) / 2
whose exact solution is:
    theta(o) = exp(-o)

We check that the neural fit reproduces the reference curve.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import frontx
import frontx.neural

jax.config.update("jax_enable_x64", True)  # better CPU precision


def test_exact() -> None:
    """The neural fit reproduces the exact exp(-o) solution within tolerance."""
    def D(theta: float | jax.Array | np.ndarray[Any, Any]) -> float | jax.Array:  # noqa: N802
        return (1 - jnp.log(theta)) / 2

    o = np.linspace(0, 20, 100)
    ref = np.exp(-o)

    sol = frontx.neural.fit(D, o, ref, i=0, b=1)

    assert sol(o) == pytest.approx(ref, abs=1e-3)
