# Tests: Van Genuchten on the validity dataset

This test fits a **VanGenuchten** diffusivity to the
`frontx.examples.data.validity` dataset using the neural API and checks the
(reduced) chi-squared. It then validates by re-solving forward with the fitted
diffusivity.

- **Fit:** `frontx.neural.fit(D, o, theta, sigma=std, i=theta_i, b=theta_b)`
- **Stat:** reduced chi-squared with `ν = len(o) − 4`

## How to run
```bash
pytest -q tests/test_vangenuchten.py
