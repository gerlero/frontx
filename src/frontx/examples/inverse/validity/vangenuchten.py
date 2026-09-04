#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

import jax

import frontx
from frontx.examples.data.validity import o, std, theta, theta_b, theta_i, theta_s
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)

plt.scatter(o, theta, label="Experimental", color="gray")

D = VanGenuchten(
    k=9.8e-14,
    m=frontx.Param(min=0, max=1),
    l=frontx.Param(min=-50, max=50),
    theta_range=(frontx.Param(min=0, max=theta_i), theta_s),
)

sol = frontx.fit(
    D,
    o,
    theta,
    i=theta_i,
    b=theta_b,
    sigma=std,
)

rchisq = np.sum((theta - sol(o)) ** 2 / std**2) / (len(o) - 4)
print("Reduced chi-squared:", rchisq)

o_display = np.linspace(0, o[-1] * 1.05, 1_000)

plt.plot(o_display, sol(o=o_display), label="frontx inverse", color="red")

sol2 = frontx.solve(
    D=sol.D,
    i=theta_i,
    b=theta_b,
)

rchisq_check = np.sum((theta - sol2(o)) ** 2 / std**2) / (len(o) - 4)
print("Reduced chi-squared (check):", rchisq_check)

plt.plot(o_display, sol2(o=o_display), label="frontx", color="blue")

plt.xlabel("o")
plt.ylabel("θ")
plt.legend()
plt.tight_layout()
plt.show()
