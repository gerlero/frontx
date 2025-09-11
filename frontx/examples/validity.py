#!/usr/bin/env python3

"""
Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
Validity of capillary imbibition models in paper-based microfluidic applications.
Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

import frontx
from frontx.examples.data.validity import o, theta, theta_i, theta_s
from frontx.models import BrooksAndCorey, LETd, LETxs, VanGenuchten

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


plt.scatter(o, theta, label="Experimental", color="gray")

epsilon = 1e-7

bc = BrooksAndCorey(n=0.2837, l=4.795, Ks=3.983e-6, theta_range=(2.378e-5, theta_s))
vg = VanGenuchten(n=8.093, l=2.344, Ks=2.079e-6, theta_range=(0.004943, theta_s))
xs = LETxs(
    Lw=1.651,
    Ew=230.5,
    Tw=0.9115,
    Ls=0.517,
    Es=493.6,
    Ts=0.3806,
    Ks=8.900e-3,
    theta_range=(0.01176, theta_s),
)
d = LETd(L=0.004569, E=12930, T=1.505, Dwt=4.660e-4, theta_range=(0.019852, theta_s))

theta_bc = frontx.solve(bc, i=theta_i, b=theta_s - epsilon)
theta_vg = frontx.solve(vg, i=theta_i, b=theta_s - epsilon)
theta_xs = frontx.solve(xs, i=theta_i, b=theta_s - epsilon)
theta_d = frontx.solve(d, i=theta_i, b=theta_s - epsilon)

o = np.linspace(0, 0.0025, 500)

plt.plot(o, theta_bc(o), label="Brooks and Corey", color="firebrick")
plt.plot(o, theta_vg(o), label="Van Genuchten", color="forestgreen")
plt.plot(o, theta_xs(o), label="LETxs", color="darkorange")
plt.plot(o, theta_d(o), label="LETd", color="dodgerblue")

plt.xlabel("o [m/√s]")
plt.ylabel("θ")
plt.legend()

plt.show()
