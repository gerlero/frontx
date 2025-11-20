# Tests: Exact (Philip, 1960)

This test verifies the classic exact case where
`D(θ) = (1 - log(θ)) / 2` ⇒ `θ(o) = exp(-o)`.  
It uses the neural API `frontx.neural.fit` to match the reference curve and
checks the absolute error.

## How to run

From the repository root:

```bash
pytest -q tests/test_exact.py
