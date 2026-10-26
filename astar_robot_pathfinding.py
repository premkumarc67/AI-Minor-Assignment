from heapq import heappush, heappop
from collections import deque
from math import hypot, inf
from pathlib import Path
import random
import time

import pandas as pd
import matplotlib.pyplot as plt


# 1. Problem configuration
ROWS = 50
COLS = 50
OBSTACLE_PROBABILITY = 0.25
NUMBER_OF_TRIALS = 100
RANDOM_SEED = 42
MIN_START_GOAL_DISTANCE = 10

OUTPUT_DIR = Path("astar_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# 2. Grid generation
MOVES = [
    (-1, 0),   # Up
    (1, 0),    # Down
    (0, -1),   # Left
    (0, 1),    # Right
]


def generate_grid(rows, cols, obstacle_probability, rng):
    """Generate a random grid.

    False = free cell
    True  = obstacle
    """
    while True:
        grid = [
            [rng.random() < obstacle_probability for _ in range(cols)]
            for _ in range(rows)
        ]

        free_cells = sum(
            not cell
            for row in grid
            for cell in row
        )

        if free_cells >= 2: # We need at least one free cell for start and one for goal.
            return grid


def get_reachable_cells(grid, start):
    """Return all free cells reachable from start using BFS."""
    rows = len(grid)
    cols = len(grid[0])

    queue = deque([start])
    visited = {start}

    while queue:
        r, c = queue.popleft()

        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc

            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and not grid[nr][nc]
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return list(visited)


def generate_solvable_scenario(
    rows,
    cols,
    obstacle_probability,
    rng,
    min_distance=10,
):
    """
    Generate:
        grid, start, goal

    The goal is selected from the connected component of the start,
    which guarantees that a solution exists.
    """
    while True:
        grid = generate_grid(
            rows,
            cols,
            obstacle_probability,
            rng,
        )

        free_cells = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if not grid[r][c]
        ]

        start = rng.choice(free_cells)
        reachable = get_reachable_cells(grid, start)

        candidates = [
            cell
            for cell in reachable
            if abs(cell[0] - start[0])
            + abs(cell[1] - start[1])
            >= min_distance
        ]

        if candidates:
            goal = rng.choice(candidates)
            return grid, start, goal


# 3. Heuristics
def h_zero(current, goal):
    """A* with h=0 becomes Dijkstra's shortest-path algorithm."""
    return 0


def h_manhattan(current, goal):
    """Manhattan distance."""
    r1, c1 = current
    r2, c2 = goal

    return abs(r1 - r2) + abs(c1 - c2)


def h_euclidean(current, goal):
    """Euclidean distance."""
    r1, c1 = current
    r2, c2 = goal

    return hypot(r1 - r2, c1 - c2)


def h_chebyshev(current, goal):
    """Chebyshev distance."""
    r1, c1 = current
    r2, c2 = goal

    return max(abs(r1 - r2), abs(c1 - c2))


HEURISTICS = {
    "Dijkstra (h=0)": h_zero,
    "Manhattan": h_manhattan,
    "Euclidean": h_euclidean,
    "Chebyshev": h_chebyshev,
}


# 4. A* implementation
def reconstruct_path(parent, goal):
    """Reconstruct the path from start to goal."""
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def astar(grid, start, goal, heuristic):
    """
    A* search.

    Returns:
        found             : whether the goal was reached
        path              : list of cells in the final path
        path_cost         : number of moves
        nodes_expanded    : number of popped/expanded states
        max_frontier      : maximum priority-queue size
        runtime_ms        : execution time in milliseconds
    """
    rows = len(grid)
    cols = len(grid[0])

    start_time = time.perf_counter()

    # Priority queue item:
    # (f = g + h, -g, insertion_counter, node)
    # Using -g makes the algorithm prefer the deeper node when f values tie. 
    # The same tie-breaking rule is used for every heuristic to keep the comparison consistent.
    open_heap = []
    insertion_counter = 0

    g_score = {start: 0} # The cheapest known cost from the start to each cell.
    parent = {start: None}
    closed = set()

    initial_f = heuristic(start, goal)
    heappush(
        open_heap,
        (initial_f, 0, insertion_counter, start)
    )

    nodes_expanded = 0
    max_frontier = 1

    while open_heap:
        f_score, negative_g, _, current = heappop(open_heap)

        current_g = g_score.get(current, inf)

        if current in closed:
            continue

        closed.add(current)
        nodes_expanded += 1

        # Goal reached.
        if current == goal:
            path = reconstruct_path(parent, goal)

            return {
                "found": True,
                "path": path,
                "path_cost": len(path) - 1,
                "nodes_expanded": nodes_expanded,
                "max_frontier": max_frontier,
                "runtime_ms": (time.perf_counter() - start_time) * 1000,
            }

        r, c = current

        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc

            # Ignore outside-grid cells.
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Ignore obstacles.
            if grid[nr][nc]:
                continue

            neighbor = (nr, nc)

            if neighbor in closed:
                continue

            tentative_g = current_g + 1

            # If this path to neighbor is better, update it.
            if tentative_g < g_score.get(neighbor, inf):
                g_score[neighbor] = tentative_g
                parent[neighbor] = current

                insertion_counter += 1

                f = (
                    tentative_g
                    + heuristic(neighbor, goal)
                )

                heappush(
                    open_heap,
                    (
                        f,
                        -tentative_g,
                        insertion_counter,
                        neighbor,
                    ),
                )

        max_frontier = max(
            max_frontier,
            len(open_heap),
        )

    # No path exists.
    return {
        "found": False,
        "path": None,
        "path_cost": inf,
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "runtime_ms": (time.perf_counter() - start_time) * 1000,
    }


# 5. Run the same scenarios for every heuristic
def run_experiment(
    number_of_trials,
    rows,
    cols,
    obstacle_probability,
    seed,
):
    """
    Generate one random scenario per trial and run every
    heuristic on that exact same scenario.
    """
    rng = random.Random(seed)
    records = []

    for trial in range(1, number_of_trials + 1):
        grid, start, goal = generate_solvable_scenario(
            rows=rows,
            cols=cols,
            obstacle_probability=obstacle_probability,
            rng=rng,
            min_distance=MIN_START_GOAL_DISTANCE,
        )

        for heuristic_name, heuristic_function in HEURISTICS.items():
            result = astar(
                grid,
                start,
                goal,
                heuristic_function,
            )

            records.append(
                {
                    "trial": trial,
                    "heuristic": heuristic_name,
                    "found": result["found"],
                    "path_cost": result["path_cost"],
                    "nodes_expanded": result["nodes_expanded"],
                    "max_frontier": result["max_frontier"],
                    "runtime_ms": result["runtime_ms"],
                }
            )

    return pd.DataFrame(records)


# 6. Summarize results
def summarize_results(results):
    summary = (
        results
        .groupby("heuristic")
        .agg(
            success_rate=("found", "mean"),
            avg_path_length=("path_cost", "mean"),
            avg_nodes_expanded=("nodes_expanded", "mean"),
            avg_max_frontier=("max_frontier", "mean"),
            avg_runtime_ms=("runtime_ms", "mean"),
        )
        .reset_index()
    )

    summary["success_rate"] *= 100

    return summary



# 8. Example grid visualization
def print_grid(grid, start, goal, path=None):
    """
    Print a grid in the terminal.

        S = start
        G = goal
        # = obstacle
        * = path
        . = free cell
    """
    path_set = set(path or [])

    for r, row in enumerate(grid):
        line = []

        for c, blocked in enumerate(row):
            cell = (r, c)

            if cell == start:
                line.append("S")
            elif cell == goal:
                line.append("G")
            elif blocked:
                line.append("#")
            elif cell in path_set:
                line.append("*")
            else:
                line.append(".")

        print(" ".join(line))


# 9. Main program
if __name__ == "__main__":

    print("=" * 70)
    print("ROBOT PATH-FINDING USING A* SEARCH")
    print("=" * 70)

    print("\nConfiguration")
    print(f"Grid size              : {ROWS} x {COLS}")
    print(f"Obstacle probability   : {OBSTACLE_PROBABILITY}")
    print(f"Number of trials       : {NUMBER_OF_TRIALS}")
    print(f"Random seed            : {RANDOM_SEED}")

    # Run experiment.
    results = run_experiment(
        number_of_trials=NUMBER_OF_TRIALS,
        rows=ROWS,
        cols=COLS,
        obstacle_probability=OBSTACLE_PROBABILITY,
        seed=RANDOM_SEED,
    )

    # Save raw results.
    csv_path = OUTPUT_DIR / "astar_results.csv"
    results.to_csv(csv_path, index=False)

    # Summary.
    summary = summarize_results(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            formatters={
                "success_rate": "{:.2f}%".format,
                "avg_path_length": "{:.2f}".format,
                "avg_nodes_expanded": "{:.2f}".format,
                "avg_max_frontier": "{:.2f}".format,
                "avg_runtime_ms": "{:.4f}".format,
            },
        )
    )

    # Save summary.
    summary_path = OUTPUT_DIR / "astar_summary.csv"
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # Show one example scenario.
    # --------------------------------------------------------
    example_rng = random.Random(RANDOM_SEED + 1000)

    grid, start, goal = generate_solvable_scenario(
        rows=20,
        cols=40,
        obstacle_probability=0.22,
        rng=example_rng,
        min_distance=15,
    )

    print("\n" + "=" * 70)
    print("EXAMPLE GRID")
    print("=" * 70)

    example_result = astar(
        grid,
        start,
        goal,
        h_manhattan,
    )

    print_grid(
        grid,
        start,
        goal,
        example_result["path"],
    )

    print("\nStart:", start)
    print("Goal :", goal)
    print("Path cost using Manhattan:", example_result["path_cost"])