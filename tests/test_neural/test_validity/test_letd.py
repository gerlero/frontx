import frontx
import frontx.neural
import jax
import numpy as np
from frontx.models import LETd

from .validity import o, std, theta, theta_b, theta_i

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def test_vangenuchten() -> None:
    """
    Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
    Validity of capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
    """

    D = LETd(  # noqa: N806
        Dwt=frontx.neural.Param(o[-1] ** 2, min=0.0),
        L=frontx.neural.Param(),
        E=frontx.neural.Param(min=0.0),
        T=frontx.neural.Param(),
        theta_range=(frontx.neural.Param(min=0.0, max=0.025), 0.7),
    )

    sol = frontx.neural.fit(D, o, theta, std, i=theta_i, b=theta_b)
    assert np.sum((sol(o) - theta) ** 2 / std**2) / (len(o) - 5) <= 0.95  # noqa: PLR2004

    sol2 = frontx.solve(sol.D, i=theta_i, b=theta_b)
    assert np.sum((sol2(o) - theta) ** 2 / std**2) / (len(o) - 5) <= 0.96  # noqa: PLR2004
