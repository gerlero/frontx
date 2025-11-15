# Example: Van Genuchten–Mualem fit and validation

This example calibrates the **VanGenuchten** model against experimental data using
the neural PINN, and validates the result with the forward solver.

- **Model:** `frontx.models.VanGenuchten` with trainable `Param`s  
- **Fit:** `frontx.neural.fit(D, o, theta, i=theta_i, b=theta_b, sigma=std)`  
- **Validation:** `frontx.solve(D=sol.D, i=theta_i, b=theta_b)`  
- **Metrics:** Reduced chi-squared reported for both predictions

## How to run

From the command line:
```bash
python -m frontx.examples.vangenuchten
