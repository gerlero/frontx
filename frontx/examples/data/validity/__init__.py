import importlib.resources

import numpy as np

"""
Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
Validity of capillary imbibition models in paper-based microfluidic applications.
Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
"""

theta_s = 0.7
theta_b = theta_s - 1e-7
theta_i = 0.025

with importlib.resources.open_binary(
    "frontx.examples.data.validity", "processed.npz"
) as p:
    _file = np.load(p)

    o = _file["o"]
    theta = theta_i + (theta_s - theta_i) * _file["I"]
    std = (theta_s - theta_i) * _file["std"]
