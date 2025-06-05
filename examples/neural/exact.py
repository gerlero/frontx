#!/usr/bin/env python3

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class Philip(eqx.Module):
    def __call__(self, theta: float | jax.Array) -> float | jax.Array:
        return (1 - jnp.log(theta)) / 2


D = Philip()

o = np.linspace(0, 20, 100)

ref = np.exp(-o)

sol = frontx.neural.fit(D, o, ref, i=0, b=1)

plt.plot(o, ref, label="Reference", color="gray")
plt.plot(o, sol(o), label="Neural fit", color="red")
plt.legend()
plt.show()
