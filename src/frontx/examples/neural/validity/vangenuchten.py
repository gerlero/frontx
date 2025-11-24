#!/usr/bin/env python3
# ruff: noqa: E402

"""Fit Van Genuchten–Mualem model with the neural PINN and validate forward.

This example calibrates :class:`frontx.models.VanGenuchten` against experimental
data using :func:`frontx.neural.fit`, and then validates by calling
:func:`frontx.solve` with the fitted diffusivity ``D``.

Run from CLI:
    python -m frontx.examples.vangenuchten
"""

import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural
from frontx.examples.data.validity import o, std, theta, theta_b, theta_i, theta_s
from frontx.models import VanGenuchten


def run() -> None:
    """Fit Van Genuchten model and plot neural vs. forward solution.

    Steps:
      1. Build a VanGenuchten model with trainable parameters (:class:`frontx.Param`).
      2. Fit with :func:`frontx.neural.fit` using experimental ``(o, theta, σ)``.
      3. Report reduced chi-squared.
      4. Re-solve with :func:`frontx.solve` using the fitted ``D`` to validate.
      5. Plot both predictions against the data.

    Returns:
      None. Displays a Matplotlib figure.
    """
    # Scatter of experimental data
    plt.scatter(o, theta, label="Experimental", color="gray")

    # Van Genuchten–Mualem model (k given, alpha/m/l trainable, bounded theta_range)
    D = VanGenuchten(
        k=9.8e-14,
        alpha=frontx.Param(min=0.0),
        m=frontx.Param(min=0.0, max=1.0),
        l=frontx.Param(),
        theta_range=(frontx.Param(min=0.0, max=theta_i), theta_s),
    )

    # Neural fit
    sol = frontx.neural.fit(
        D,
        o,
        theta,
        i=theta_i,
        b=theta_b,
        sigma=std,
    )

    # Goodness of fit (reduced chi-squared)
    rchisq = np.sum((theta - sol(o)) ** 2 / std**2) / (len(o) - 4)
    print("Reduced chi-squared:", rchisq)  # noqa: T201

    # Smooth display grid
    o_display = np.linspace(0, o[-1] * 1.05, 1_000)

    # Plot neural fit
    plt.plot(o_display, sol(o=o_display), label="frontx.neural", color="red")

    # Forward-solver validation with fitted D
    sol2 = frontx.solve(
        D=sol.D,
        i=theta_i,
        b=theta_b,
    )

    rchisq_check = np.sum((theta - sol2(o)) ** 2 / std**2) / (len(o) - 4)
    print("Reduced chi-squared (check):", rchisq_check)  # noqa: T201

    plt.plot(o_display, sol2(o=o_display), label="frontx", color="blue")

    plt.xlabel("o")
    plt.ylabel("θ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
