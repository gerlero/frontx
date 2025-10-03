"""Van Genuchten fit against experimental validity dataset.

Fits a VanGenuchten–Mualem diffusivity with trainable parameters to the
`frontx.examples.data.validity` dataset using the neural API, then checks the
(reduced) chi-squared. It also validates by re-solving forward with the fitted
D and re-evaluating the statistic.

Reference:
    Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
    Validity of capillary imbibition models in paper-based microfluidic applications.
    Transport in Porous Media, 141(2), 359–378. https://doi.org/10.1007/s11242-021-01724-w
"""

import numpy as np

import frontx
import frontx.neural
from frontx.examples.data.validity import o, std, theta, theta_b, theta_i, theta_s
from frontx.models import VanGenuchten


def test_vangenuchten() -> None:
    """Neural fit of VanGenuchten and forward validation on the validity dataset."""
    D = VanGenuchten(  # noqa: N806
        k=9.8e-14,
        alpha=frontx.Param(min=0.0),
        m=frontx.Param(min=0.0, max=1.0),
        l=frontx.Param(),
        theta_range=(frontx.Param(min=0.0, max=theta_i), theta_s),
    )

    # Neural fit
    sol = frontx.neural.fit(D, o, theta, std, i=theta_i, b=theta_b)
    rchisq = np.sum((sol(o) - theta) ** 2 / std**2) / (len(o) - 4)
    assert rchisq <= 2.5  # empirical threshold from the study

    # Forward re-solve with the fitted D and re-check
    sol2 = frontx.solve(sol.D, i=theta_i, b=theta_b)
    rchisq2 = np.sum((sol2(o) - theta) ** 2 / std**2) / (len(o) - 4)
    assert rchisq2 <= 2.9
