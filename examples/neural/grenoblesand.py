#!/usr/bin/env python3

import jax
import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

Ks = 15.37  # cm/h
alpha = 0.0432  # 1/cm
m = 0.5096
theta_s = 0.312

D = VanGenuchten(Ks=Ks, alpha=alpha, m=m, theta_range=(0.0, theta_s))

ref = frontx.solve(
    D,
    i=0,
    b=theta_s - 1e-7,
)

D = VanGenuchten(
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

print(f"Ks/alpha={sol.D.Ks.value}, m={sol.D.m.value}")  # noqa: T201

o = np.linspace(0, ref.oi * 1.5, 500)

plt.plot(o, ref(o), label="Reference", color="gray")
plt.plot(o, sol(o), label="Neural fit", color="red")
plt.legend()
plt.show()
