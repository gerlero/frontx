"""Physics-Informed Neural Network (PINN) utilities for Frontx.

This module provides the public API to train a PINN and to evaluate the learned
solution. The internal training network is implementation detail; consumers
should use the :class:`Solution` class (immutable wrapper around the trained
network) and the :func:`fit` routine to obtain it.

The PINN models a scalar mapping ``o -> theta(o)`` constrained by data and a
physics residual defined via a diffusivity-like callable ``D``.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._boltzmann import AbstractSolution, boltzmannmethod
from ._util import vmap


class _PINN(eqx.Module):
    D: Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]
    _net: eqx.nn.MLP = eqx.field(
        default_factory=lambda: eqx.nn.MLP(
            "scalar",
            "scalar",
            8,
            3,
            jax.nn.tanh,
            key=jax.random.PRNGKey(1199692436803635474),  # = hash("frontx"))
        )
    )

    def __call__(
        self, x: float | jax.Array | np.ndarray[Any, Any]
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        return 2 * jax.nn.sigmoid(-x * jax.nn.softplus(vmap(self._net)(x)))

    def data_loss(
        self,
        x_data: jax.Array | np.ndarray[Any, Any],
        y_data: jax.Array | np.ndarray[Any, Any],
        /,
        y_sigma: float | jax.Array | np.ndarray[Any, Any] = 1,
    ) -> jax.Array:
        return jnp.mean(((self(x_data) - y_data) / (y_sigma)) ** 2)

    def physics_residuals(self, /, *, i: float, b: float, oi: float) -> jax.Array:
        x = jnp.linspace(0, 1, 501)[1:]

        lhs = -x / 2 * vmap(jax.grad(self))(x)
        rhs = (
            vmap(
                jax.grad(
                    lambda x: self.D(i + (b - i) * self(x)) * vmap(jax.grad(self))(x)
                )
            )(x)
            / oi**2
        )

        return lhs - rhs

    def physics_loss(
        self, /, *, i: float, b: float, oi: float, residual_cutoff: float = jnp.inf
    ) -> jax.Array:
        residuals = self.physics_residuals(i=i, b=b, oi=oi)

        residuals = jnp.where(
            jnp.abs(residuals) > residual_cutoff,
            jnp.nan,
            residuals,
        )

        return jnp.nanmean(residuals**2) + self(1.0) ** 2 + jax.grad(self)(1.0) ** 2


class Solution(AbstractSolution):
    """Trained PINN solution wrapper.

    This class exposes a callable solution ``theta(o)`` together with its
    configuration. The underlying neural network and training logic are
    internal; users typically obtain an instance via :func:`fit`.

    The solution maps an observable/coordinate ``o`` to a scalar response
    constrained to lie between ``i`` and ``b`` after a normalized transform.
    The scaling parameter ``oi`` provides a characteristic range for ``o`` and
    is used to normalize inputs during training and inference.

    Attributes:
        oi: Characteristic scale used to normalize inputs ``o`` (``x = o/oi``).
    """

    oi: float
    _net: _PINN
    _i: float
    _b: float

    def __init__(
        self,
        net: _PINN,
        *,
        i: float,
        b: float,
        oi: float,
    ) -> None:
        """Initialize a :class:`Solution`.

        Args:
            net: Trained internal PINN network (implementation detail).
            i: Lower bound (baseline) of the response range.
            b: Upper bound (saturation) of the response range.
            oi: Characteristic input scale used for normalization.
        """
        self._net = net
        self._i = i
        self._b = b
        self.oi = oi
        super().__init__()

    @property
    def D(  # noqa: N802
        self,
    ) -> Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]:
        """Return the diffusivity-like callable used in the physics term.

        Returns:
            A callable that accepts a scalar/array and returns a scalar/array
            with the same broadcastable shape. It is the same function that was
            passed to :func:`fit` as ``D``.
        """
        return self._net.D

    @boltzmannmethod
    def __call__(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
        """Evaluate the trained solution at input ``o``.

        The input is clipped to ``[0, oi]`` (after normalization) and the
        network prediction is rescaled back to the original range ``[i, b]``.

        Args:
            o: Input coordinate(s) at which to evaluate the solution.

        Returns:
            The predicted response with the same shape broadcasting as ``o``.
        """
        x = jnp.clip(o / self.oi, 0, 1)

        return self._i + (self._b - self._i) * self._net(x)


@eqx.filter_jit
def fit(  # noqa: PLR0913
    D: Callable[  # noqa: N803
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ],
    o: jax.Array | np.ndarray[Any, Any],
    theta: jax.Array | np.ndarray[Any, Any],
    /,
    sigma: float | jax.Array | np.ndarray[Any, Any] | None = None,
    *,
    i: float,
    b: float,
    oi: float | None = None,
    max_steps: int = 300_000,
) -> Solution:
    """Train a PINN against data and physics, returning a callable solution.

    This routine fits a physics-informed neural network using paired data
    ``(o, theta)`` and a physics residual weighted by an annealed coefficient.
    Inputs are normalized by ``oi`` and outputs are scaled to ``[i, b]``.
    Training stops early when the physics loss target is reached or when
    ``max_steps`` is exceeded.

    Args:
        D: Diffusivity-like callable used in the physics residual.
        o: Input coordinates (1D array-like). Typically monotonically increasing.
        theta: Target responses aligned with ``o``.
        sigma: Optional observational noise (scalar or array-like) to weight the
            data loss. If ``None``, unit weight is used.
        i: Lower bound (baseline) of the response range.
        b: Upper bound (saturation) of the response range.
        oi: Optional input scale. If ``None``, it defaults to ``1.05 * o[-1]``.
        max_steps: Maximum number of optimization steps.

    Returns:
        A :class:`Solution` object wrapping the trained network.

    Raises:
        RuntimeError: If the physics loss fails to converge to the target
            threshold before ``max_steps`` (raised by ``eqx.error_if``).
        AssertionError: If internal normalization parameters are missing.
    """
    if oi is None:
        oi = o[-1] * 1.05  # ty: ignore [invalid-assignment]

    assert oi is not None

    net = _PINN(D)

    x_data = o / oi
    y_data = (theta - i) / (b - i)
    y_sigma = sigma / (b - i) if sigma is not None else 1

    initial_data_loss = net.data_loss(x_data, y_data, y_sigma=y_sigma)

    trainable_net, static_net = eqx.partition(net, eqx.is_array)

    optim = optax.adam(
        learning_rate=optax.exponential_decay(1e-3, 100_000, 0.1, end_value=1e-4)
    )
    opt_state = optim.init(trainable_net)

    def loss(
        trainable_net: _PINN, *, step: int, residual_cutoff: float = jnp.inf
    ) -> jax.Array:
        net = eqx.combine(trainable_net, static_net)

        assert oi is not None
        physics_loss = net.physics_loss(
            i=i, b=b, oi=oi, residual_cutoff=residual_cutoff
        )

        data_loss = net.data_loss(x_data, y_data, y_sigma=y_sigma)

        lambda_ = initial_data_loss * 10 ** (-2 + step / 100_000)

        return data_loss + lambda_ * physics_loss

    def train_step(
        trainable_net: _PINN,
        opt_state: optax.OptState,
        step: int,
        physics_loss: float,
        residual_cutoff: float,
    ) -> tuple[_PINN, optax.OptState, int, float, float]:
        net = eqx.combine(trainable_net, static_net)

        assert oi is not None
        residuals = net.physics_residuals(i=i, b=b, oi=oi)
        spike_score = (
            jnp.max(jnp.abs(residuals)) - jnp.percentile(jnp.abs(residuals), 99)
        ) / (jnp.median(jnp.abs(jnp.abs(residuals) - jnp.median(jnp.abs(residuals)))))
        residual_cutoff = jax.lax.select(  # ty: ignore [invalid-assignment]
            (step >= 50_000) & (residual_cutoff == jnp.inf) & (spike_score > 200),  # noqa: PLR2004
            jnp.mean(jnp.abs(residuals)),
            residual_cutoff,
        )

        physics_loss = net.physics_loss(  # ty: ignore [invalid-assignment]
            i=i, b=b, oi=oi, residual_cutoff=residual_cutoff
        )

        grads = jax.grad(loss)(
            trainable_net,
            step=step,
            residual_cutoff=residual_cutoff,
        )

        updates, opt_state = optim.update(grads, opt_state)

        trainable_net = eqx.apply_updates(trainable_net, updates)

        return trainable_net, opt_state, step + 1, physics_loss, residual_cutoff

    physics_loss_target = 1e-5

    trainable_net, opt_state, _, physics_loss, _ = jax.lax.while_loop(
        lambda val: (val[2] <= max_steps) & (val[3] > physics_loss_target),
        lambda val: train_step(*val),
        (
            trainable_net,
            opt_state,
            0,
            jnp.inf,
            jnp.inf,
        ),
    )

    trainable_net = eqx.error_if(
        trainable_net,
        physics_loss > physics_loss_target,
        "Physics loss did not converge",
    )

    return Solution(
        net=eqx.combine(trainable_net, static_net),
        i=i,
        b=b,
        oi=oi,
    )
