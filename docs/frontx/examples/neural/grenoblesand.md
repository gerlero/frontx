# Example: GrenobleSand (synthetic Van Genuchten → neural fit)

This example generates synthetic data from a **VanGenuchten** model (“Grenoble sand”
parameters), then fits a neural PINN to recover the parameters and compares
reference vs. prediction.

- **Reference model:** `VanGenuchten(Ks, alpha, m, theta_range=(0, θs))`
- **Fit model:** `VanGenuchten(Ks=Param, m=Param, theta_range=(0, θs))`
- **Workflow:** forward solve → neural fit → plot comparison

## How to run

From the command line:
```bash
python -m frontx.examples.grenoblesand
