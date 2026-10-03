#!/usr/bin/env python3
import time

import jax
import matplotlib.pyplot as plt
import numpy as np

import frontx
from frontx.models import BrooksAndCorey, VanGenuchten

jax.config.update("jax_enable_x64", True)

# https://doi.org/10.1016/0022-1694(92)90032-Q
Ks = 15.37  # cm/h
alpha_vg = 0.0432  # 1/cm
m_vg = 0.5096
Ks_bc = 4.27
alpha_bc = 1 / 11.43  # 1/cm
n_bc = 1.2876
theta_s = 0.312

vg = VanGenuchten(Ks=Ks, alpha=alpha_vg, m=m_vg, theta_range=(0.0, theta_s))
sol_vg = frontx.solve(vg, i=0, b=theta_s - 1e-7)

jax.block_until_ready(sol_vg)
start_time = time.perf_counter()
jax.block_until_ready(frontx.solve(vg, i=0, b=theta_s - 1e-7))
print(f"Time to solve (Van Genuchten): {time.perf_counter() - start_time:.3f} s")

bc = BrooksAndCorey(Ks=Ks, alpha=alpha_bc, n=n_bc, theta_range=(0.0, theta_s))
sol_bc = frontx.solve(bc, i=0, b=theta_s - 1e-7)

jax.block_until_ready(sol_bc)
start_time = time.perf_counter()
jax.block_until_ready(frontx.solve(bc, i=0, b=theta_s - 1e-7))
print(f"Time to solve (Brooks and Corey): {time.perf_counter() - start_time:.3f} s")

o_display = np.linspace(0, sol_vg.oi * 1.5, 500)
plt.plot(o_display, sol_vg(o_display), label="Van Genuchten")
plt.plot(o_display, sol_bc(o_display), label="Brooks and Corey")
plt.xlabel("o [cm/√h]")
plt.ylabel("θ")
plt.legend()
plt.tight_layout()
plt.show()
