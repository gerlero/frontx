import equinox as eqx
import frontx
import frontx.neural
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def test_exact() -> None:
    """
    Philip, J. R. (1960). General Method of Exact Solution of the
    Concentration-Dependent Diffusion Equation.
    Australian Journal of Physics, 13(1), 1-12. https://doi.org/10.1071/PH600001
    """

    class Philip(eqx.Module):
        def __call__(self, theta: jax.Array) -> jax.Array:
            return (1 - jnp.log(theta)) / 2

    o = np.linspace(0, 20, 100)

    ref = np.exp(-o)

    sol = frontx.neural.fit(Philip(), o, ref, i=0, b=1)

    assert sol(o) == pytest.approx(ref, abs=1e-3)
