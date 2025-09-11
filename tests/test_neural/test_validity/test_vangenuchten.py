import frontx
import frontx.neural
import numpy as np
from frontx.examples.data.validity import o, std, theta, theta_b, theta_i, theta_s
from frontx.models import VanGenuchten


def test_vangenuchten() -> None:
    """
    Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
    Validity of capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
    """

    D = VanGenuchten(  # noqa: N806
        k=9.8e-14,
        alpha=frontx.Param(min=0.0),
        m=frontx.Param(min=0.0, max=1.0),
        l=frontx.Param(),
        theta_range=(frontx.Param(min=0.0, max=theta_i), theta_s),
    )

    sol = frontx.neural.fit(D, o, theta, std, i=theta_i, b=theta_b)
    assert np.sum((sol(o) - theta) ** 2 / std**2) / (len(o) - 4) <= 2.5  # noqa: PLR2004

    sol2 = frontx.solve(sol.D, i=theta_i, b=theta_b)
    assert np.sum((sol2(o) - theta) ** 2 / std**2) / (len(o) - 4) <= 2.9  # noqa: PLR2004
