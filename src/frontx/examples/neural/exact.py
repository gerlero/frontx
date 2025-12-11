#!/usr/bin/env python3

"""ExactI: simple neural fit against a known reference curve.

This example fits the public PINN API (:func:`frontx.neural.fit`) to a synthetic
dataset where the target is known in closed form, to check that the learned
solution reproduces it.

The model uses a custom diffusivity ``D(theta) = (1 - log(theta)) / 2`` and
fits ``theta(o)`` in the interval ``o ∈ [0, 20]`` against the reference curve
``exp(-o)`` with bounds ``i=0`` and ``b=1``.

Run from CLI:
    python -m frontx.examples.exacti
"""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import frontx
import frontx.neural

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def D(theta: float | jax.Array | np.ndarray[Any, Any]) -> float | jax.Array:  # noqa: N802
    """Custom diffusivity used by the PINN.

    Args:
        theta: State value(s).

    Returns:
        The diffusivity evaluated at ``theta`` with broadcastable shape.
    """
    return (1 - jnp.log(theta)) / 2


def run() -> None:
    """Fit the PINN to the exact reference and plot the result.

    Generates a 1D grid ``o``, computes the reference response ``exp(-o)``,
    trains the neural solution with :func:`frontx.neural.fit`, and plots
    reference vs. prediction.
    """
    o = np.linspace(0, 20, 100)
    ref = np.exp(-o)

    sol = frontx.neural.fit(D, o, ref, i=0, b=1)

    plt.plot(o, ref, label="Reference", color="gray")
    plt.plot(o, sol(o), label="Neural fit", color="red")
    plt.xlabel("o")
    plt.ylabel("θ")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
