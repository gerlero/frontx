from ._forward import RESULTS, Solution, solve
from ._inverse import InterpolatedSolution, ScaledSolution, sorptivity

__all__ = [
    "RESULTS",
    "InterpolatedSolution",
    "ScaledSolution",
    "Solution",
    "solve",
    "sorptivity",
]
