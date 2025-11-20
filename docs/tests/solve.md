# Tests: `frontx.solve`

This suite exercises the forward solver along three fronts:

1. **Exact case (Philip, 1960):** with `D(θ) = (1 − log(θ)) / 2`, the solution is
   `θ(o) = exp(-o)`.
2. **External reference comparison (`fronts`):**
   for LETd, LETxs, VanGenuchten, and Brooks–Corey models it checks:
   - agreement of `D`, `dD/dθ`, and `d²D/dθ²` with `fronts.D`,
   - agreement between `frontx.solve` and `fronts.solve` (abs ≤ 5e-2).
3. **Unsolvable cases:** the solver must raise or flag failure
   (`result != successful`) under impossible conditions.

## How to run

From the project root:

```bash
pytest -q tests/test_solve.py
