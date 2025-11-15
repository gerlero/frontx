# Tests: Grenoble Sand (synthetic → neural fit)

This test generates synthetic data with a **VanGenuchten** model (Grenoble-sand
parameters) and checks that fitting with `frontx.neural.fit` reproduces the
reference curve.

- **Reference:** `VanGenuchten(Ks, alpha, m, theta_range=(0, θs))`
- **Fit:** `VanGenuchten(Ks=Param, m=Param, theta_range=(0, θs))`
- **Check:** `|θ_pred(o) − θ_ref(o)| ≤ 1e-3` for `o ∈ [0, oi_ref]`

## How to run

```bash
pytest -q tests/test_grenoblesand.py
