from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from typing import Generic, TypeVar

from hardy_ls.move import Move
from hardy_ls.neighborhood import Neighborhood
from hardy_ls.objective import Objective
from hardy_ls.solution import Solution
from hardy_ls.types import Direction, ImprovementStrategy

S = TypeVar("S", bound=Solution)


@dataclass
class Result(Generic[S]):
    """Search result payload."""

    best_solution: S
    best_value: float
    solution_history: list[S]
    move_history: list[Move[S]]
    num_trials: int


class LocalSolver(Generic[S]):
    """
    Generic local search solver.

    Problem-specific pieces:
      - concrete `Solution`
      - concrete `Move`
      - `Objective.evaluate`
      - one or more neighborhoods
    """

    def __init__(
        self,
        objective: Objective[S],
        neighborhoods: list[Neighborhood[S]],
        *,
        strategy: ImprovementStrategy = "first",
        direction: Direction = "min",
    ) -> None:
        if strategy not in ("first", "best"):
            raise ValueError("strategy must be 'first' or 'best'")
        if direction not in ("min", "max"):
            raise ValueError("direction must be 'min' or 'max'")

        self._objective = objective
        self._neighborhoods = neighborhoods
        self._strategy = strategy
        self._direction = direction

    def solve(
        self,
        initial_solution: S,
        time_limit: timedelta | None = None,
        *,
        max_trials: int | None = None,
    ) -> Result[S]:
        """
        Run local search and return the best solution found.

        At least one stop condition must be set: `time_limit` or `max_trials`.
        """

        if time_limit is None and max_trials is None:
            raise ValueError("Either time_limit or max_trials must be specified")

        start_time = perf_counter()

        current = initial_solution.clone()
        current_value = self._objective.evaluate(current)

        move_history: list[Move[S]] = []

        best = current.clone()
        best_value = current_value

        num_trials = 0

        while True:
            elapsed_time = perf_counter() - start_time

            if (
                (time_limit is not None and elapsed_time >= time_limit.total_seconds())
                or max_trials is not None
                and num_trials >= max_trials
            ):
                break

            iteration_improved = False

            for neighborhood in self._neighborhoods:
                best_delta: float | None = None
                best_move: Move[S] | None = None

                for move in neighborhood.generate(current):
                    num_trials += 1

                    delta = neighborhood.delta(self._objective, current, move)

                    if not self._is_improving(delta):
                        continue

                    if self._strategy == "first":
                        candidate = current.clone()
                        move.apply(candidate)
                        current = candidate
                        current_value += delta
                        move_history.append(move)
                        iteration_improved = True
                        break

                    # strategy == "best"
                    if best_delta is None or self._is_better_delta(delta, best_delta):
                        best_delta = delta
                        best_move = move

                if (
                    self._strategy == "best"
                    and best_delta is not None
                    and best_move is not None
                ):
                    candidate = current.clone()
                    best_move.apply(candidate)
                    current = candidate
                    current_value += best_delta
                    move_history.append(best_move)
                    iteration_improved = True

                if iteration_improved:
                    if self._is_better_value(current_value, best_value):
                        best = current.clone()
                        best_value = current_value
                    break

            if not iteration_improved:
                # No improving move across all neighborhoods: local optimum reached.
                break

        return Result(
            best_solution=best,
            best_value=best_value,
            solution_history=self._restore_solution_history(
                initial_solution, move_history
            ),
            move_history=move_history,
            num_trials=num_trials,
        )

    @staticmethod
    def _restore_solution_history(
        initial_solution: S, move_history: list[Move[S]]
    ) -> list[S]:
        solution = initial_solution.clone()
        solution_history = [solution]
        for move in move_history:
            move.apply(solution)
            solution_history.append(solution.clone())
        return solution_history

    def _is_improving(self, delta: float) -> bool:
        return delta < 0 if self._direction == "min" else delta > 0

    def _is_better_delta(self, lhs: float, rhs: float) -> bool:
        return lhs < rhs if self._direction == "min" else lhs > rhs

    def _is_better_value(self, lhs: float, rhs: float) -> bool:
        return lhs < rhs if self._direction == "min" else lhs > rhs
