import frontx
import jax.numpy as jnp
import pytest


def test_exact() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    o = jnp.linspace(0, 20, 100)

    sol = frontx.InterpolatedSolution(o, jnp.exp(-o))

    assert sol(o) == pytest.approx(jnp.exp(-o))

    theta = jnp.linspace(1e-6, 1, 100)

    assert sol.D(theta) == pytest.approx(0.5 * (1 - jnp.log(theta)), abs=5e-2)


def test_exact_solve() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    o = jnp.linspace(0, 20, 100)

    sol = frontx.InterpolatedSolution(o, jnp.exp(-o))

    theta = frontx.solve(D=sol.D, b=1, i=1e-3, itol=5e-3)

    assert theta(o=o) == pytest.approx(jnp.exp(-o), abs=1e-2)


def test_sorptivity() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    o = jnp.linspace(0, 20, 100)

    sol = frontx.InterpolatedSolution(o, jnp.exp(-o))

    assert sol.sorptivity() == pytest.approx(1, abs=1e-4)
