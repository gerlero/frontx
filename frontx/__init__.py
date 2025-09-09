from ._forward import RESULTS, Solution, solve
from ._inverse import InterpolatedSolution, Param, ScaledSolution, sorptivity

__all__ = [
    "RESULTS",
    "InterpolatedSolution",
    "Param",
    "ScaledSolution",
    "Solution",
    "solve",
    "sorptivity",
]
