from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self


@dataclass
class Solution(ABC):
    """Problem-specific solution state."""

    @abstractmethod
    def clone(self) -> Self:
        """Return an independent copy of this solution."""
