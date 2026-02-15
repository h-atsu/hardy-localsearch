from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from hardy_ls.solution import Solution

S = TypeVar("S", bound=Solution)


class Objective(ABC, Generic[S]):
    """Objective function interface (minimization by default in solver)."""

    @abstractmethod
    def evaluate(self, sol: S) -> float:
        """Return the objective value for the given solution."""
