# Example: ExactI (neural fit to a known reference)

This example fits the PINN to a synthetic target to verify that the learned
solution reproduces a known curve.

- **Diffusivity:** `D(theta) = (1 - log(theta)) / 2`  
- **Domain:** `o ∈ [0, 20]`  
- **Reference:** `theta_ref(o) = exp(-o)`  
- **Bounds:** `i = 0`, `b = 1`

## How to run

From the command line:
```bash
python -m frontx.examples.exacti
