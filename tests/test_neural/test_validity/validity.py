from pathlib import Path

import numpy as np

"""
Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).
Validity of capillary imbibition models in paper-based microfluidic applications.
Transport in Porous Media, 141(2), 359-378. https://doi.org/10.1007/s11242-021-01724-w
"""

theta_s = 0.7
theta_b = theta_s - 1e-7
theta_i = 0.025

_file = np.load(Path(__file__).parent / "processed.npz")
o = _file["o"]
theta = theta_i + (theta_s - theta_i) * _file["I"]
std = (theta_s - theta_i) * _file["std"]
