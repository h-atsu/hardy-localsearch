from __future__ import annotations

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Self


@dataclass
class Solution(ABC):
    """Problem-specific solution state."""

    def clone(self) -> Self:
        """Return an independent copy of this solution."""
        return deepcopy(self)
