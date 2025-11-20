# Tests: `frontx.finite`

This suite validates two core properties of the finite-difference solver:

1. **Consistency** with the reference solver (`frontx.solve`) using LET-family
   models (LETd, LETxs).
2. **Mass conservation:** the spatial integral of `θ(r, t) − i` matches the
   **sorptivity** reported by the reference solution.

## How to run

From the repository root:

```bash
pytest -q
