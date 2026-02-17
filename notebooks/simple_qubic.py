from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

from hardy_ls import LocalSolver, Move, Neighborhood, Objective, Solution


@dataclass
class IntSolution(Solution):
    x: int


@dataclass(frozen=True)
class PlusOne(Move[IntSolution]):
    def apply(self, sol: IntSolution) -> None:
        sol.x += 1


@dataclass(frozen=True)
class MinusOne(Move[IntSolution]):
    def apply(self, sol: IntSolution) -> None:
        sol.x -= 1


class QuadraticObjective(Objective[IntSolution]):
    # Minimize f(x) = (x - 3)^2
    def evaluate(self, sol: IntSolution) -> float:
        return float((sol.x - 3) ** 2)


class StepNeighborhood(Neighborhood[IntSolution]):
    def generate(self, solution: IntSolution) -> Iterable[Move[IntSolution]]:
        yield PlusOne()
        yield MinusOne()


solver = LocalSolver(
    objective=QuadraticObjective(),
    neighborhoods=[StepNeighborhood()],
    strategy="best",
    direction="min",
)

result = solver.solve(
    initial_solution=IntSolution(10),
    time_limit=timedelta(seconds=1),
    max_trials=10_000,
)

print(result.best_solution, result.best_value, result.num_trials)
