"""Public package exports for hardy-ls."""

from hardy_ls.move import Move
from hardy_ls.neighborhood import Neighborhood
from hardy_ls.objective import Objective
from hardy_ls.solution import Solution
from hardy_ls.solver import (
    LocalSolver,
    Result,
)

__all__ = [
    "LocalSolver",
    "Result",
    "Move",
    "Neighborhood",
    "Objective",
    "Solution",
]


def main() -> None:
    """Entry point for the `hardy-ls` console script."""
    print("hardy-ls provides a local-search library API. See README.md for usage.")
