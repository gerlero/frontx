#!/usr/bin/env python3
import time

import jax
import matplotlib.pyplot as plt
import numpy as np

import frontx
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)

Ks = 15.37  # cm/h
alpha = 0.0432  # 1/cm
m = 0.5096
theta_s = 0.312

D = VanGenuchten(Ks=Ks, alpha=alpha, m=m, theta_range=(0.0, theta_s))
sol = frontx.solve(D, i=0, b=theta_s - 1e-7)

jax.block_until_ready(sol)
start_time = time.perf_counter()
jax.block_until_ready(frontx.solve(D, i=0, b=theta_s - 1e-7))
print(f"Time to solve: {time.perf_counter() - start_time:.3f} s")

o_display = np.linspace(0, sol.oi * 1.5, 500)
plt.plot(o_display, sol(o_display))
plt.xlabel("o")
plt.ylabel("θ")
plt.tight_layout()
plt.show()
