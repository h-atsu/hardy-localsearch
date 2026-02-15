from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

from hardy_ls.move import Move
from hardy_ls.objective import Objective
from hardy_ls.solution import Solution

S = TypeVar("S", bound=Solution)


class Neighborhood(ABC, Generic[S]):
    """Neighborhood that generates candidate moves for a solution."""

    @abstractmethod
    def generate(self, solution: S) -> Iterable[Move[S]]:
        """Yield moves around the given solution."""

    def delta(self, objective: Objective[S], sol: S, move: Move[S]) -> float:
        """Return `objective(new_solution) - objective(old_solution)`."""
        candidate = sol.clone()
        move.apply(candidate)
        delta = objective.evaluate(candidate) - objective.evaluate(sol)
        return delta

    @property
    def has_custom_delta(self) -> bool:
        """Whether this neighborhood overrides `delta`."""
        return type(self).delta is not Neighborhood.delta
