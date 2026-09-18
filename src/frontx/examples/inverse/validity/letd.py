#!/usr/bin/env python3

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

import frontx
from frontx.examples.data.validity import o, std, theta, theta_b, theta_i, theta_s
from frontx.models import LETd

jax.config.update("jax_enable_x64", True)


def solve(args):
    L, E, T, theta_r = args
    D = LETd(Dwt=1e-3, L=L, E=E, T=T, theta_range=(theta_r, theta_s))
    sol = frontx.solve(D=D, i=theta_i, b=theta_b, throw=False)
    return frontx.ScaledSolution.fitting_data(sol, o, theta, std, throw=False)


@jax.jit
def cost(args):
    sol = solve(args)
    return jnp.sum((theta - sol(o)) ** 2 / std**2) / (len(o) - 5)


start_time = time.perf_counter()
opt = differential_evolution(
    cost,
    bounds=[(0, 10), (0, 1e5), (0, 10), (0.0, theta_i)],
    disp=True,
    polish=False,
    atol=1,
    seed=42,
)
print("Inverse solve time:", time.perf_counter() - start_time, "seconds")

sol = solve(opt.x)
rchisq = jnp.sum((theta - sol(o)) ** 2 / std**2) / (len(o) - 5)
print("Reduced chi-squared:", rchisq)

sol2 = frontx.solve(
    D=sol.D,
    i=theta_i,
    b=theta_b,
)

rchisq_check = jnp.sum((theta - sol2(o)) ** 2 / std**2) / (len(o) - 5)
print("Reduced chi-squared (check):", rchisq_check)

o_display = jnp.linspace(0, o[-1] * 1.05, 1_000)

plt.scatter(o, theta, label="Experimental", color="gray")
plt.plot(o_display, sol(o=o_display), label="Inverse", color="red")
plt.plot(o_display, sol2(o=o_display), label="Check", color="blue")
plt.xlabel("o")
plt.ylabel("θ")
plt.legend()
plt.tight_layout()
plt.show()
