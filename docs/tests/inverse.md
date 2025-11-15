# Tests: `frontx._inverse` / parameter fitting

This suite checks that inversion/fitting methods recover coherent parameters or
models from data (real or synthetic) and that results are **consistent** with
the forward solver.

## What is covered

1. **Parameter recovery** in diffusivity models (e.g., VanGenuchten, LETd):
   the fit reproduces reference values within tolerances.
2. **Forward/inverse consistency:** when the fitted model is used in
   `frontx.solve`, it predicts the same response obtained during fitting.
3. **Noise robustness:** fitting remains stable with `sigma ≠ 0` and respects
   bounds via `Param(min=..., max=...)`.
4. **Reasonable convergence:** e.g., bounded number of iterations/steps.

## How to run

From the repository root:

```bash
pytest -q
