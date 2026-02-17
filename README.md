# hardy-ls

`hardy-ls` is a lightweight framework for implementing local search (neighborhood search) for combinatorial optimization problems.  
You define problem-specific `Solution`, `Move`, `Objective`, and `Neighborhood` classes, then run the search with `LocalSolver`.

## Current Scope

- Unconstrained objective optimization (`min` / `max`)
- Improvement strategies: `first` / `best`
- Multiple neighborhoods can be configured and applied in order

## Not Supported Yet

- **Constrained optimization problems are not supported yet**
  - There is currently no built-in mechanism for feasibility checks, repair operators, or penalty-based constraint handling.
- **Kick/perturbation for escaping local optima is not implemented yet**
  - There is no built-in ILS-style kick step or restart controller in the solver API.

## Installation

### With `uv`

```bash
uv sync
```

### With `pip`

```bash
pip install -e .
```

## Minimal Example

```python
from dataclasses import dataclass
from datetime import timedelta
from typing import Self, Iterable

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
```

## Core API

- `Solution`: solution state (`clone()` required)
- `Move`: in-place operation on a solution (`apply()` required)
- `Objective`: objective function (`evaluate()` required)
- `Neighborhood`: move generator (`generate()` required)
- `LocalSolver`: local search engine

## Notes

- A full usage example is available in `notebooks/tsp.py`.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
