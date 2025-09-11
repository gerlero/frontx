from typing import Any

import frontx
import frontx.neural
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_exact() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """

    def D(theta: float | jax.Array | np.ndarray[Any, Any]) -> float | jax.Array:  # noqa: N802
        return (1 - jnp.log(theta)) / 2

    o = np.linspace(0, 20, 100)

    ref = np.exp(-o)

    sol = frontx.neural.fit(D, o, ref, i=0, b=1)

    assert sol(o) == pytest.approx(ref, abs=1e-3)
