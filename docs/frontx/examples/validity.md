# Example: Validity of capillary imbibition models

This example reproduces the comparison from:

> Gerlero, G. S., Valdez, A. R., Urteaga, R., & Kler, P. A. (2022).  
> *Validity of capillary imbibition models in paper-based microfluidic applications*.  
> Transport in Porous Media, 141(2), 359–378. https://doi.org/10.1007/s11242-021-01724-w

It compares four models (`BrooksAndCorey`, `VanGenuchten`, `LETxs`, `LETd`) against experimental data using the public API `frontx.solve`.

## How to run

From the command line:
```bash
python -m frontx.examples.validity
