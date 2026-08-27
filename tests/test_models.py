import jax
import jax.numpy as jnp
import pytest

from frontx.models import VanGenuchten

jax.config.update("jax_enable_x64", True)


def test_van_genuchten_near_residual() -> None:
    D = VanGenuchten(n=1.1)

    # Reference values computed at high precision.
    theta = 1e-3
    expected = (
        2.613452611709472e-36,
        3.0054705034658917e-32,
        3.1557440286391852e-28,
    )

    assert D(theta) == pytest.approx(expected[0], rel=1e-10, abs=0.0)
    assert jax.grad(D)(theta) == pytest.approx(expected[1], rel=1e-10, abs=0.0)
    assert jax.grad(jax.grad(D))(theta) == pytest.approx(
        expected[2], rel=1e-10, abs=0.0
    )


def test_van_genuchten_kr_near_residual_float32() -> None:
    model = VanGenuchten(n=1.5)

    theta = jnp.asarray(1e-3, dtype=jnp.float32)

    assert model._kr(theta) == pytest.approx(
        3.5136418469739605e-21,
        rel=1e-6,
        abs=0.0,
    )


def test_van_genuchten_diffusivity() -> None:
    model = VanGenuchten(n=1.5, theta_range=(0.1, 0.9))

    theta = jnp.linspace(0.1 + 1e-9, 0.9 - 1e-9, 100)

    def K_over_C(theta):
        return model._K(theta) / model._C(theta)

    assert jax.vmap(model)(theta) == pytest.approx(jax.vmap(K_over_C)(theta))
    assert jax.vmap(jax.grad(model))(theta) == pytest.approx(
        jax.vmap(jax.grad(K_over_C))(theta)
    )
    assert jax.vmap(jax.grad(jax.grad(model)))(theta) == pytest.approx(
        jax.vmap(jax.grad(jax.grad(K_over_C)))(theta)
    )
