#!/usr/bin/env python3

"""Neural fit on synthetic data generated from Van Genuchten (Grenoble sand).

This example:
  1) Define a Van Genuchten diffusivity with known parameters
     (representative “Grenoble sand” values).
  2) Generate a *reference* solution with :func:`frontx.solve`.
  3) Fit a neural PINN (:func:`frontx.neural.fit`) where only a subset of
     parameters is trainable (e.g., Ks and m), using the synthetic reference.
  4) Compare reference vs. neural prediction.

Run from CLI:
    python -m frontx.examples.grenoblesand
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def run() -> None:
    """Generate synthetic data and fit a neural PINN; plot the comparison."""
    # Known parameters (units as in the original setup)
    Ks = 15.37  # cm/h  # noqa: N806
    alpha = 0.0432  # 1/cm
    m = 0.5096
    theta_s = 0.312

    # Reference model and forward solution (nearly saturated boundary)
    D_ref = VanGenuchten(Ks=Ks, alpha=alpha, m=m, theta_range=(0.0, theta_s))  # noqa: N806
    ref = frontx.solve(D_ref, i=0, b=theta_s - 1e-7)

    # Trainable model (subset of parameters to infer from synthetic data)
    D_train = VanGenuchten(  # noqa: N806
        Ks=frontx.Param(min=0.0),
        m=frontx.Param(min=0.0, max=1.0),
        theta_range=(0.0, theta_s),
    )

    # Fit on synthetic observations from the reference
    o = np.linspace(0, ref.oi, 100)
    sol = frontx.neural.fit(
        D_train,
        o,
        ref(o),
        i=0,
        b=theta_s - 1e-7,
    )

    print(f"Ks={sol.D.Ks.value}, m={sol.D.m.value}")  # noqa: T201

    # Display
    o_display = np.linspace(0, ref.oi * 1.5, 500)
    plt.plot(o_display, ref(o_display), label="Reference", color="gray")
    plt.plot(o_display, sol(o_display), label="Neural fit", color="red")
    plt.xlabel("o")
    plt.ylabel("θ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
