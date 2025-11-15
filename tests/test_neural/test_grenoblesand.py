"""Synthetic Grenoble Sand test for the neural PINN.

Generates synthetic data with a Van Genuchten–Mualem model (Grenoble-sand-like
parameters) and checks that the neural fit reproduces the reference curve within
a tight absolute tolerance.

Reference:
    Fuentes, C., Haverkamp, R., & Parlange, J.-Y. (1992).
    Parameter constraints on closed-form soil-water relationships.
    Journal of Hydrology, 134(1–4), 117–142. https://doi.org/10.1016/0022-1694(92)90032-Q
"""

import jax
import numpy as np
import pytest

import frontx
import frontx.neural
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)  # better CPU precision


def test_grenoble_sand() -> None:
    """The PINN fits synthetic Van Genuchten data (Grenoble sand)."""
    Ks = 15.37       # cm/h
    alpha = 0.0432   # 1/cm
    m = 0.5096
    theta_s = 0.312

    # Reference model and nearly saturated boundary forward solution
    D_ref = VanGenuchten(Ks=Ks, alpha=alpha, m=m, theta_range=(0.0, theta_s))
    ref = frontx.solve(D_ref, i=0, b=theta_s - 1e-7)

    # Trainable model: infer Ks and m from the synthetic reference
    D_train = VanGenuchten(
        Ks=frontx.Param(min=0.0),
        m=frontx.Param(min=0.0, max=1.0),
        theta_range=(0.0, theta_s),
    )

    # Neural fit on the reference curve
    o = np.linspace(0, ref.oi, 100)
    sol = frontx.neural.fit(D_train, o, ref(o), i=0, b=theta_s - 1e-7)

    # Quantitative comparison
    assert sol(o) == pytest.approx(ref(o), abs=1e-3)
