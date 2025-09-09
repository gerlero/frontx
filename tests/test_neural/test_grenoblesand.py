import frontx
import frontx.neural
import jax
import numpy as np
import pytest
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def test_grenoble_sand() -> None:
    """
    Fuentes, C., Haverkamp, R., Parlange, J.-Y. (1992). Parameter constraints on
    closed-form soilwater relationships.
    Journal of Hydrology, 134(1-4), 117-142. https://doi.org/10.1016/0022-1694(92)90032-Q
    """
    Ks = 15.37  # cm/h  # noqa: N806
    alpha = 0.0432  # 1/cm
    m = 0.5096
    theta_s = 0.312

    D = VanGenuchten(Ks=Ks, alpha=alpha, m=m, theta_range=(0.0, theta_s))  # noqa: N806

    ref = frontx.solve(
        D,
        i=0,
        b=theta_s - 1e-7,
    )

    D = VanGenuchten(  # noqa: N806
        Ks=frontx.Param(min=0.0),
        m=frontx.Param(min=0.0, max=1.0),
        theta_range=(0.0, theta_s),
    )

    o = np.linspace(0, ref.oi, 100)

    sol = frontx.neural.fit(
        D,
        o,
        ref(o),
        i=0,
        b=theta_s - 1e-7,
    )

    assert sol(o) == pytest.approx(ref(o), abs=1e-3)
