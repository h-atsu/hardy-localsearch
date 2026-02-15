from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from hardy_ls.solution import Solution

S = TypeVar("S", bound=Solution)


class Move(ABC, Generic[S]):
    """Problem-specific move operation."""

    name: str

    @abstractmethod
    def apply(self, sol: S) -> None:
        """Apply this move in-place to the given solution."""
