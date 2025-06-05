#!/usr/bin/env python3

import jax
import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural
from frontx.models import LETd
from validity import (  # ty: ignore [unresolved-import]
    o,
    std,
    theta,
    theta_b,
    theta_i,
    theta_s,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

plt.scatter(o, theta, label="Experimental", color="gray")

D = LETd(
    Dwt=frontx.neural.Param(o[-1] ** 2, min=0.0),
    L=frontx.neural.Param(),
    E=frontx.neural.Param(min=0.0),
    T=frontx.neural.Param(),
    theta_range=(frontx.neural.Param(min=0.0, max=theta_i), theta_s),
)

sol = frontx.neural.fit(
    D,
    o,
    theta,
    i=theta_i,
    b=theta_b,
    sigma=std,
)

rchisq = np.sum((theta - sol(o)) ** 2 / std**2) / (len(o) - 5)
print("Reduced chi-squared:", rchisq)  # noqa: T201

o_display = np.linspace(0, o[-1] * 1.05, 1_000)

plt.plot(o_display, sol(o=o_display), label="frontx.neural", color="red")

sol2 = frontx.solve(
    D=sol.D,
    i=theta_i,
    b=theta_b,
)

rchisq = np.sum((theta - sol2(o)) ** 2 / std**2) / (len(o) - 5)
print("Reduced chi-squared (check):", rchisq)  # noqa: T201

plt.plot(o_display, sol2(o=o_display), label="frontx", color="blue")

plt.legend()
plt.show()
