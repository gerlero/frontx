import jax
import jax.numpy as jnp
import parametrix as pmx


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
