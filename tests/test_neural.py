import frontx.neural
import jax.numpy as jnp
import numpy as np
import pytest


def test_solve() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    theta = frontx.neural.solve(D=lambda theta: (1 - jnp.log(theta)) / 2, i=0, b=1)

    o = np.linspace(0, 20, 100)

    assert theta(o) == pytest.approx(np.exp(-o), abs=1e-3)
