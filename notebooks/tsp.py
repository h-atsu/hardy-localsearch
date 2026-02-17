# %%

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hardy_ls import (
    LocalSolver,
    Move,
    Neighborhood,
    Objective,
    Solution,
)


# %%
@dataclass
class TSPSolution(Solution):
    route: np.ndarray


@dataclass(frozen=True)
class TwoOptMove(Move[TSPSolution]):
    i: int
    j: int

    def apply(self, sol: TSPSolution) -> None:
        sol.route[self.i : self.j + 1] = sol.route[self.i : self.j + 1][::-1]


class TSPObjective(Objective[TSPSolution]):
    def __init__(self, dist: np.ndarray) -> None:
        self.dist = dist

    def evaluate(self, sol: TSPSolution) -> float:
        route = sol.route
        nxt = np.roll(route, -1)
        return float(np.sum(self.dist[route, nxt]))


@dataclass(frozen=True)
class OrOptMove(Move[TSPSolution]):
    i: int
    after: int

    def apply(self, sol: TSPSolution) -> None:
        route = sol.route
        node = route[self.i]
        reduced = np.delete(route, self.i)
        insert_pos = self.after + 1 if self.after < self.i else self.after
        sol.route = np.insert(reduced, insert_pos, node)


class TwoOptNeighborhood(Neighborhood[TSPSolution]):
    name = "two_opt"

    def generate(self, solution: TSPSolution) -> Iterable[Move[TSPSolution]]:
        n = len(solution.route)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                yield TwoOptMove(i=i, j=j)

    def delta(
        self,
        objective: Objective[TSPSolution],
        sol: TSPSolution,
        move: Move[TSPSolution],
    ) -> float:
        if not isinstance(move, TwoOptMove) or not isinstance(objective, TSPObjective):
            return super().delta(objective, sol, move)

        n = len(sol.route)
        a = sol.route[(move.i - 1) % n]
        b = sol.route[move.i]
        c = sol.route[move.j]
        d = sol.route[(move.j + 1) % n]
        return (
            objective.dist[a, c]
            + objective.dist[b, d]
            - objective.dist[a, b]
            - objective.dist[c, d]
        )


class OrOptNeighborhood(Neighborhood[TSPSolution]):
    name = "or_opt_1"

    def generate(self, solution: TSPSolution) -> Iterable[Move[TSPSolution]]:
        n = len(solution.route)
        for i in range(n):
            for after in range(n):
                if after == i or after == (i - 1) % n:
                    continue
                yield OrOptMove(i=i, after=after)

    def delta(
        self,
        objective: Objective[TSPSolution],
        sol: TSPSolution,
        move: Move[TSPSolution],
    ) -> float:
        if not isinstance(move, OrOptMove) or not isinstance(objective, TSPObjective):
            return super().delta(objective, sol, move)

        n = len(sol.route)
        route = sol.route
        a = route[(move.i - 1) % n]
        x = route[move.i]
        b = route[(move.i + 1) % n]
        y = route[move.after]
        z = route[(move.after + 1) % n]
        removed = objective.dist[a, x] + objective.dist[x, b] + objective.dist[y, z]
        added = objective.dist[a, b] + objective.dist[y, x] + objective.dist[x, z]
        return added - removed


def make_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def closed_route(route: np.ndarray) -> np.ndarray:
    return np.concatenate([route, route[:1]])


def make_tour_trace(
    coords: np.ndarray, route: np.ndarray, name: str, color: str
) -> go.Scatter:
    cyc = closed_route(route)
    return go.Scatter(
        x=coords[cyc, 0],
        y=coords[cyc, 1],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=2),
        marker=dict(size=6),
    )


def plot_routes(
    coords: np.ndarray, initial_route: np.ndarray, best_route: np.ndarray
) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Initial Route", "Best Route"))
    fig.add_trace(
        make_tour_trace(coords, initial_route, "Initial", "#e76f51"),
        row=1,
        col=1,
    )
    fig.add_trace(
        make_tour_trace(coords, best_route, "Best", "#2a9d8f"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="TSP Route Comparison",
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def running_min(values: list[float]) -> list[float]:
    out: list[float] = []
    best = float("inf")
    for v in values:
        best = min(best, v)
        out.append(best)
    return out


def plot_ils_convergence(restart_best_values: list[float]) -> go.Figure:
    global_best = running_min(restart_best_values)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=restart_best_values,
            mode="lines+markers",
            name="Restart local best",
            line=dict(color="#577590", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=global_best,
            mode="lines+markers",
            name="Global best so far",
            line=dict(color="#264653", width=2),
        )
    )
    fig.update_layout(
        title="ILS Convergence",
        xaxis_title="Restart",
        yaxis_title="Tour Length",
        height=420,
    )
    return fig


def safe_show(fig: go.Figure) -> None:
    try:
        fig.show()
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped] {type(exc).__name__}: {exc}")


# %%
rng_np = np.random.default_rng(42)
n_cities = 50
coords: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = rng_np.uniform(
    0.0, 100.0, size=(n_cities, 2)
)
dist = make_distance_matrix(coords)

initial_route = rng_np.permutation(n_cities)
initial_solution = TSPSolution(route=initial_route)
objective = TSPObjective(dist)

local_solver = LocalSolver(
    objective=objective,
    neighborhoods=[
        TwoOptNeighborhood(),
        OrOptNeighborhood(),
    ],
    strategy="best",
    direction="min",
)


result = local_solver.solve(
    initial_solution=initial_solution,
    time_limit=timedelta(seconds=10),
)

print("=== TSP Local Solver Demo ===")
print(f"initial length : {objective.evaluate(initial_solution):.4f}")
print(f"best length    : {result.best_value:.4f}")
print(f"total trials   : {result.num_trials}")
print(f"best route     : {result.best_solution.route.tolist()}")

fig_routes = plot_routes(coords, initial_route, result.best_solution.route)

safe_show(fig_routes)
# %%
