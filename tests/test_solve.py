from typing import Any

import fronts
import fronts.D
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from frontx import solve
from frontx.models import BrooksAndCorey, LETd, LETxs, VanGenuchten

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]  # noqa: FBT003


def test_exact() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """
    theta = solve(D=lambda theta: (1 - jnp.log(theta)) / 2, i=0, b=1)

    o = np.linspace(0, 20, 100)

    assert theta(o) == pytest.approx(np.exp(-o), abs=1e-3)


@pytest.mark.parametrize(
    "Ds",
    [
        (
            LETd,
            fronts.D.letd,
            {
                "L": 0.004569,
                "E": 12930,
                "T": 1.505,
                "Dwt": 4.660e-4,
                "theta_range": (0.019852, 0.7),
            },
        ),
        (
            LETd,
            fronts.D.letd,
            {
                "Dwt": 1.004e-3,
                "L": 1.356,
                "E": 10010,
                "T": 1.224,
                "theta_range": (0.00625, 0.7),
            },
        ),
        (
            VanGenuchten,
            fronts.D.van_genuchten,
            {
                "n": 8.093,
                "l": 2.344,
                "Ks": 2.079e-6,
                "theta_range": (0.004943, 0.7),
            },
        ),
        (
            VanGenuchten,
            fronts.D.van_genuchten,
            {
                "m": 0.8861,
                "l": 2.331,
                "Ks": 2.105e-6,
                "theta_range": (0.005623, 0.7),
            },
        ),
        (
            BrooksAndCorey,
            fronts.D.brooks_and_corey,
            {
                "n": 0.2837,
                "l": 4.795,
                "Ks": 3.983e-6,
                "theta_range": (2.378e-5, 0.7),
            },
        ),
        (
            LETxs,
            fronts.D.letxs,
            {
                "Lw": 1.651,
                "Ew": 230.5,
                "Tw": 0.9115,
                "Ls": 0.517,
                "Es": 493.6,
                "Ts": 0.3806,
                "Ks": 8.900e-3,
                "theta_range": (0.01176, 0.7),
            },
        ),
    ],
)
def test_fronts_papers(Ds: tuple[Any, Any, dict[str, Any]]) -> None:  # noqa: N803
    """
    Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
    Validity of capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w

    Gerlero, G. S., Berli, C. L. A., & Kler, P. A. (2023).
    Open-source high-performance software packages for direct and inverse solving of
    horizontal capillary flow.
    Capillarity, 6(2), 31-40. https://doi.org/10.46690/capi.2023.02.02
    """
    D, Df, kwargs = Ds  # noqa: N806
    D = D(**kwargs)  # noqa: N806
    Df = Df(**kwargs)  # noqa: N806

    b = 0.7 - 1e-7
    i = 0.025

    assert D(0.5) == pytest.approx(Df(0.5))
    assert D(i) == pytest.approx(Df(i))

    assert jax.grad(D)(0.5) == pytest.approx(Df(0.5, 1)[1])
    assert jax.grad(D)(i) == pytest.approx(Df(i, 1)[1])

    assert jax.grad(jax.grad(D))(0.5) == pytest.approx(Df(0.5, 2)[2])
    assert jax.grad(jax.grad(D))(i) == pytest.approx(Df(i, 2)[2])

    solf = fronts.solve(Df, b=b, i=i)
    sol = solve(D, b=b, i=i)

    o = np.linspace(0, solf.oi, 100)

    assert sol(o) == pytest.approx(solf(o=o), abs=5e-2)
