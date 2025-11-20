#!/usr/bin/env python3
# ruff: noqa: E402

"""Fit LETd model with the neural PINN and validate with the forward solver.

This example calibrates the empirical :class:`frontx.models.LETd` diffusivity
against experimental data using :func:`frontx.neural.fit`. It then validates the
result by solving again with :func:`frontx.solve` using the fitted ``D``.

The dataset is loaded from :mod:`frontx.examples.data.validity` and the plot
shows raw measurements with standard deviation, the neural fit, and the
forward-solver check.

Run from CLI:
    python -m frontx.examples.letd
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural
from frontx.examples.data.validity import o, std, theta, theta_b, theta_i, theta_s
from frontx.models import LETd


def run() -> None:
    """Fit LETd to data and plot neural vs. forward solution.

    Steps:
      1. Build an LETd model with trainable parameters (:class:`frontx.Param`).
      2. Fit with :func:`frontx.neural.fit` using experimental ``(o, theta, σ)``.
      3. Report reduced chi-squared.
      4. Re-solve with :func:`frontx.solve` using the fitted ``D`` to validate.
      5. Plot both predictions against the data.

    Returns:
      None. Displays a Matplotlib figure.
    """
    # Scatter of experimental data
    plt.scatter(o, theta, label="Experimental", color="gray")

    # LETd model with trainable parameters and bounds
    D = LETd(
        Dwt=frontx.Param(o[-1] ** 2, min=0.0),
        L=frontx.Param(),
        E=frontx.Param(min=0.0),
        T=frontx.Param(),
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
    rchisq = np.sum((theta - sol(o)) ** 2 / std**2) / (len(o) - 5)
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

    rchisq_check = np.sum((theta - sol2(o)) ** 2 / std**2) / (len(o) - 5)
    print("Reduced chi-squared (check):", rchisq_check)  # noqa: T201

    plt.plot(o_display, sol2(o=o_display), label="frontx", color="blue")

    plt.xlabel("o")
    plt.ylabel("θ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
