import frontx
import jax.numpy as jnp
import pytest
from frontx.models import LETd


def test_inteprolated_exact() -> None:
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


def test_interpolated_exact_solve() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    o = jnp.linspace(0, 20, 100)

    sol = frontx.InterpolatedSolution(o, jnp.exp(-o))

    theta = frontx.solve(D=sol.D, b=1, i=1e-3, itol=5e-3)

    assert theta(o=o) == pytest.approx(jnp.exp(-o), abs=1e-2)


def test_interpolated_sorptivity() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    o = jnp.linspace(0, 20, 100)

    sol = frontx.InterpolatedSolution(o, jnp.exp(-o))

    assert sol.sorptivity() == pytest.approx(1, abs=1e-4)


def test_standalone_sorptivity() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    o = jnp.linspace(0, 20, 500)
    assert frontx.sorptivity(o, jnp.exp(-o), b=1, i=0) == pytest.approx(1, abs=1e-3)


def test_scaled_solution_sorptivity() -> None:
    """
    Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
    Validity of capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
    """
    D = LETd(L=0.004569, E=12930, T=1.505, Dwt=4.660e-4, theta_range=(0.019852, 0.7))  # noqa: N806
    b = 0.7 - 1e-7
    i = 0.025

    sol = frontx.solve(D, b=b, i=i)

    unscaled = frontx.ScaledSolution(sol, D0=1 / D.Dwt)

    scaled = frontx.ScaledSolution.with_sorptivity(unscaled, sol.sorptivity())

    assert scaled.sorptivity() == pytest.approx(sol.sorptivity())

    assert scaled.D0 == pytest.approx(D.Dwt)  # noqa: SIM300
    assert scaled.oi == pytest.approx(sol.oi)


def test_scaled_solution_data() -> None:
    """
    Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
    Validity of capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
    """
    D = LETd(L=0.004569, E=12930, T=1.505, Dwt=4.660e-4, theta_range=(0.019852, 0.7))  # noqa: N806
    b = 0.7 - 1e-7
    i = 0.025

    sol = frontx.solve(D, b=b, i=i)

    unscaled = frontx.ScaledSolution(sol, D0=1 / D.Dwt)

    o = jnp.linspace(0, sol.oi, 500)

    scaled = frontx.ScaledSolution.fitting_data(unscaled, o, sol(o))

    assert scaled(o) == pytest.approx(sol(o), abs=1e-6)
    assert scaled.d_do(o) == pytest.approx(sol.d_do(o), abs=1e-2)

    assert scaled.D0 == pytest.approx(D.Dwt)  # noqa: SIM300
    assert scaled.sorptivity() == pytest.approx(sol.sorptivity())
    assert scaled.oi == pytest.approx(sol.oi)
