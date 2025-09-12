from collections.abc import Callable, Sequence
from typing import Any, TypeVar, overload

import jax
import jax.numpy as jnp
import numpy as np
import parametrix as pmx
from scipy.optimize import differential_evolution


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


def get_params(pytree: object, /) -> Sequence[Param]:
    ret: list[Param] = []

    def collect(obj: object) -> None:
        if isinstance(obj, Param):
            ret.append(obj)

    jax.tree_util.tree_map(
        collect,
        pytree,
        is_leaf=lambda obj: isinstance(obj, Param),
    )
    return ret


_T = TypeVar("_T")
_O = TypeVar("_O")


def set_param_values(
    pytree: _T, values: jax.Array | np.ndarray[Any, Any] | Sequence[float], /
) -> _T:
    i = 0

    @overload
    def replace(obj: Param) -> Param: ...

    @overload
    def replace(obj: _O) -> _O: ...

    def replace(obj: Param | _O) -> Param | _O:
        nonlocal i
        if isinstance(obj, Param):
            obj = Param(values[i], min=obj.min, max=obj.max)  # ty: ignore [invalid-argument-type]
            i += 1
        return obj

    return jax.tree_util.tree_map(
        replace, pytree, is_leaf=lambda obj: isinstance(obj, Param)
    )


def de_fit(
    candidate: Callable[[_T], _O],
    cost: Callable[[_O], float],
    /,
    initial: _T,
    *,
    max_steps: int = 15,
) -> _O:
    params = get_params(initial)
    x0 = jnp.array([p.value for p in params])
    bounds = jnp.array([(p.min, p.max) for p in params])

    opt = differential_evolution(
        jax.jit(lambda x: cost(candidate(set_param_values(initial, x)))),
        bounds=bounds,
        x0=x0,
        maxiter=max_steps,
    )

    return candidate(set_param_values(initial, opt.x))
