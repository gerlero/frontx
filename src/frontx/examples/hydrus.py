#!/usr/bin/env python3
import time

import jax
import matplotlib.pyplot as plt

import frontx
from frontx.examples.data.hydrus import r, t, theta, velocity
from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)

Ks = 25  # cm/h
alpha = 0.01433  # 1/cm
n = 1.506
theta_range = (0, 0.3308)

theta_i = 0.1003
theta_b = theta_range[1] - 1e-7

D = VanGenuchten(Ks=Ks, alpha=alpha, n=n, theta_range=theta_range)
sol = frontx.solve(D, i=theta_i, b=theta_b)
jax.block_until_ready(sol)

start_time = time.perf_counter()
jax.block_until_ready(frontx.solve(D, i=theta_i, b=theta_b))
print(f"Time to solve: {time.perf_counter() - start_time:.3f} s")

plt.figure()
plt.title("Water content fields")
for t_, theta_ in zip(t, theta, strict=True):
    plt.plot(r, theta_, label=f"Hydrus-1D, t={t_} h")
    plt.plot(r, sol(r, t_), label=f"Frontx, t={t_} h", linestyle="--")
plt.xlabel("r [cm]")
plt.ylabel("θ")
plt.grid(which="both")
plt.legend()

plt.figure()
plt.title("Velocity fields")
for t_, velocity_ in zip(t, velocity, strict=True):
    plt.plot(r, velocity_, label=f"Hydrus-1D, t={t_} h")
    plt.plot(r, sol.flux(r, t_), label=f"Frontx, t={t_} h", linestyle="--")
plt.xlabel("r [cm]")
plt.ylabel("u [cm/h]")
plt.grid(which="both")
plt.legend()

plt.show()
