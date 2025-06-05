from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import parametrix as pmx

from ._boltzmann import AbstractSolution, boltzmannmethod
from ._util import vmap


class Param(pmx.Param[float]):
    min: float | None
    max: float | None

    def __init__(
        self,
        value: float | jax.Array | None = None,
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
    ) -> None:
        if min is not None and max is not None:
            if value is None:
                value = (min + max) / 2
            value = jnp.log((value - min) / (max - value))
        elif min is not None:
            if value is None:
                value = min + 1.0
            value = jnp.log(value - min)
        elif max is not None:
            if value is None:
                value = max - 1.0
            value = jnp.log(max - value)
        elif value is None:
            value = 0.0

        super().__init__(value)
        self.min = min
        self.max = max

    @property
    def value(self) -> jax.Array:
        if self.min is not None and self.max is not None:
            return self.min + (self.max - self.min) * jax.nn.sigmoid(self.raw_value)
        if self.min is not None:
            return self.min + jnp.exp(self.raw_value)
        if self.max is not None:
            return self.max - jnp.exp(self.raw_value)
        return self.raw_value


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

    def physics_loss(
        self,
        /,
        *,
        i: float,
        b: float,
        oi: float,
        step: int,
    ) -> jax.Array:
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

        residuals = lhs - rhs

        residuals = jnp.where(
            (step >= 30_000) & (jnp.abs(residuals) > jnp.mean(jnp.abs(residuals))),
            jnp.nan,
            residuals,
        )

        return jnp.nanmean(residuals**2) + self(1.0) ** 2 + jax.grad(self)(1.0) ** 2


class Solution(AbstractSolution):
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
        self._net = net
        self._i = i
        self._b = b
        self.oi = oi
        super().__init__()

    @property
    def D(
        self,
    ) -> Callable[
        [float | jax.Array | np.ndarray[Any, Any]],
        float | jax.Array | np.ndarray[Any, Any],
    ]:
        return self._net.D

    @boltzmannmethod
    def __call__(
        self,
        o: float | jax.Array | np.ndarray[Any, Any],
    ) -> float | jax.Array | np.ndarray[Any, Any]:
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
    max_steps: int = 100_000,
) -> Solution:
    if oi is None:
        oi = o[-1] * 1.05

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

    def loss(trainable_net: _PINN, *, step: int) -> jax.Array:
        net = eqx.combine(trainable_net, static_net)

        physics_loss = net.physics_loss(i=i, b=b, oi=oi, step=step)

        data_loss = net.data_loss(x_data, y_data, y_sigma=y_sigma)

        lambda_ = initial_data_loss * 10 ** (-2 + step / 100_000)

        return data_loss + lambda_ * physics_loss, physics_loss

    def adam_train_step(
        trainable_net: _PINN,
        opt_state: optax.OptState,
        physics_loss: jax.Array,
        step: int,
    ) -> tuple[_PINN, optax.OptState]:
        (_, physics_loss), grads = jax.value_and_grad(loss, has_aux=True)(
            trainable_net, step=step
        )

        updates, opt_state = optim.update(grads, opt_state)

        trainable_net = eqx.apply_updates(trainable_net, updates)

        return trainable_net, opt_state, physics_loss, step + 1

    trainable_net, opt_state, physics_loss, step = jax.lax.while_loop(
        lambda val: (val[2] > 1e-5) & (val[3] < max_steps),
        lambda val: adam_train_step(*val),
        (trainable_net, opt_state, jnp.inf, 0),
    )

    net = eqx.combine(trainable_net, static_net)

    return Solution(
        net,
        i=i,
        b=b,
        oi=oi,
    )
