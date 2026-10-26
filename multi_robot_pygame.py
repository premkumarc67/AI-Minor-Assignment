"""Interactive multi-robot path finding demo using A* and CBA*.

Controls are available in the Pygame side panel, so you do not need to edit
constants to change the number of robots, grid size, or obstacle percentage.

All robots are scheduled on a shared clock and appear on their own start cells
at t=0. In the independent-A* baseline they may collide; CBA* attempts to find
collision-free paths using space-time constraints and wait actions.

Run:
    pip install pygame
    python multi_robot_pygame.py
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, inf
import random
import time

# Pygame is imported inside run_app() so the path-planning functions can still
# be imported and tested on machines where Pygame has not been installed.

# ----------------------------- Defaults ------------------------------------
DEFAULT_ROWS = 18           # grid height Y (number of rows)
DEFAULT_COLS = 24           # grid width X (number of columns)
DEFAULT_ROBOTS = 5
DEFAULT_OBSTACLE_PERCENT = 20
MAX_HIGH_LEVEL_EXPANSIONS = 180
MIN_START_GOAL_DISTANCE = 4

# Screen layout
SCREEN_W, SCREEN_H = 1400, 900
GRID_AREA = (28, 112, 925, 748)
PANEL_X = 975

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WAIT = (0, 0)
SPACE_TIME_MOVES = MOVES + [WAIT]

AGENT_COLOURS = [
    (52, 152, 219), (231, 76, 60), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (236, 112, 99),
    (127, 140, 141), (52, 73, 94), (243, 156, 18), (22, 160, 133),
]


@dataclass(frozen=True)
class AgentScenario:
    start: tuple[int, int]
    goal: tuple[int, int]


@dataclass
class AgentConstraints:
    """Hard space-time constraints for one robot."""
    vertex: set = field(default_factory=set)  # (time, cell)
    edge: set = field(default_factory=set)    # (time, from_cell, to_cell)

    def copy(self) -> "AgentConstraints":
        return AgentConstraints(set(self.vertex), set(self.edge))

    def signature(self):
        return (tuple(sorted(self.vertex)), tuple(sorted(self.edge)))


# ------------------------- Grid / scenario ---------------------------------
def neighbours(grid, cell):
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc]:
            yield nr, nc


def connected_components(grid):
    rows, cols = len(grid), len(grid[0])
    seen = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            start = (r, c)
            if grid[r][c] or start in seen:
                continue
            component = []
            queue = deque([start])
            seen.add(start)
            while queue:
                cell = queue.popleft()
                component.append(cell)
                for nxt in neighbours(grid, cell):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            components.append(component)
    return components


def generate_multi_robot_scenario(rows: int, cols: int, obstacle_percent: int,
                                  number_of_agents: int, rng: random.Random):
    """Create an exact-count obstacle grid and unique reachable start/goal pairs.

    All starts and goals are unique, and all endpoints are picked from one
    connected component. This avoids impossible tasks caused by disconnected
    start-goal pairs. The actual obstacle count is round(rows*cols*pct/100).
    """
    if rows * cols < 2 * number_of_agents:
        raise ValueError("The grid needs at least two distinct endpoint cells per robot.")

    total_cells = rows * cols
    obstacle_count = round(total_cells * obstacle_percent / 100)
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    min_distance = max(3, min(rows, cols) // 3)

    for _attempt in range(350):
        shuffled = all_cells[:]
        rng.shuffle(shuffled)
        blocked = set(shuffled[:obstacle_count])
        grid = [[(r, c) in blocked for c in range(cols)] for r in range(rows)]
        components = connected_components(grid)
        component = max(components, key=len, default=[])
        if len(component) < 2 * number_of_agents:
            continue

        available = component[:]
        rng.shuffle(available)
        agents = []
        success = True
        for _ in range(number_of_agents):
            # Pick a start that still has at least one sufficiently distant goal.
            chosen_start = None
            possible_goals = None
            for start in available:
                goals = [p for p in available if p != start and
                         abs(p[0] - start[0]) + abs(p[1] - start[1]) >= min_distance]
                if goals:
                    chosen_start = start
                    possible_goals = goals
                    break
            if chosen_start is None or not possible_goals:
                success = False
                break
            goal = rng.choice(possible_goals)
            agents.append(AgentScenario(chosen_start, goal))
            available.remove(chosen_start)
            available.remove(goal)

        if success and len(agents) == number_of_agents:
            return grid, agents

    raise RuntimeError(
        "Could not generate a connected scenario for these settings. "
        "Try a larger grid, fewer robots, or fewer obstacles."
    )


# ------------------------------- A* -----------------------------------------
def h_manhattan(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def h_euclidean(current, goal):
    return hypot(current[0] - goal[0], current[1] - goal[1])


def reconstruct_path(parent, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def astar_single(grid, start, goal, heuristic=h_manhattan):
    """Plain A* for one robot, ignoring the other robots."""
    open_heap = []
    counter = 0
    g_score = {start: 0}
    parent = {start: None}
    closed = set()
    heappush(open_heap, (heuristic(start, goal), 0, counter, start))
    nodes_expanded = 0

    while open_heap:
        _, _, _, current = heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(parent, goal)
            return {"found": True, "path": path, "cost": len(path) - 1,
                    "nodes_expanded": nodes_expanded}

        current_g = g_score[current]
        for nxt in neighbours(grid, current):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(nxt, inf):
                g_score[nxt] = tentative_g
                parent[nxt] = current
                counter += 1
                heappush(open_heap, (tentative_g + heuristic(nxt, goal),
                                     -tentative_g, counter, nxt))

    return {"found": False, "path": None, "cost": inf,
            "nodes_expanded": nodes_expanded}


def position_at(path, t):
    """Robots stay at their goal after arriving, until the shared simulation ends."""
    return path[t] if t < len(path) else path[-1]


def detect_collisions(paths):
    """Count vertex and edge-swap collision events across synchronized time."""
    if not paths:
        return {"collision_events": 0, "vertex_collisions": 0,
                "edge_swap_collisions": 0, "collision_pairs": 0}
    makespan = max(len(p) - 1 for p in paths)
    vertex = edge = 0
    pairs = set()
    for t in range(makespan + 1):
        occupancy = defaultdict(list)
        for i, path in enumerate(paths):
            occupancy[position_at(path, t)].append(i)
        for ids in occupancy.values():
            if len(ids) > 1:
                vertex += len(ids) * (len(ids) - 1) // 2
                for k in range(len(ids)):
                    for j in range(k + 1, len(ids)):
                        pairs.add(tuple(sorted((ids[k], ids[j]))))
        if t < makespan:
            for i in range(len(paths)):
                a0, a1 = position_at(paths[i], t), position_at(paths[i], t + 1)
                for j in range(i + 1, len(paths)):
                    b0, b1 = position_at(paths[j], t), position_at(paths[j], t + 1)
                    if a0 == b1 and a1 == b0 and a0 != a1:
                        edge += 1
                        pairs.add((i, j))
    return {"collision_events": vertex + edge, "vertex_collisions": vertex,
            "edge_swap_collisions": edge, "collision_pairs": len(pairs)}


def independent_astar(grid, agents):
    start_time = time.perf_counter()
    paths, nodes = [], 0
    for agent in agents:
        result = astar_single(grid, agent.start, agent.goal)
        nodes += result["nodes_expanded"]
        if not result["found"]:
            return {"success": False, "paths": None, "total_nodes_expanded": nodes,
                    "planning_time_ms": (time.perf_counter() - start_time) * 1000,
                    **detect_collisions([])}
        paths.append(result["path"])
    collision_stats = detect_collisions(paths)
    return {
        "success": collision_stats["collision_events"] == 0,
        "paths": paths,
        "makespan": max(len(p) - 1 for p in paths),
        "sum_of_costs": sum(len(p) - 1 for p in paths),
        "total_nodes_expanded": nodes,
        "planning_time_ms": (time.perf_counter() - start_time) * 1000,
        "wait_steps": sum(1 for p in paths for a, b in zip(p, p[1:]) if a == b),
        "high_level_expansions": 0,
        **collision_stats,
    }


# ------------------- Proposed method: adaptive CBA* -------------------------
def local_obstacle_density(grid, cell, radius=1):
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    total = blocked = 0
    for rr in range(max(0, r-radius), min(rows, r+radius+1)):
        for cc in range(max(0, c-radius), min(cols, c+radius+1)):
            total += 1
            blocked += int(grid[rr][cc])
    return blocked / total if total else 0.0


def _all_conflicts(grid, paths):
    if not paths:
        return []
    makespan = max(len(p) - 1 for p in paths)
    conflicts = []
    pair_frequency = defaultdict(int)
    for t in range(makespan + 1):
        for i in range(len(paths)):
            pi = position_at(paths[i], t)
            for j in range(i + 1, len(paths)):
                pj = position_at(paths[j], t)
                if pi == pj:
                    pair_frequency[(i, j)] += 1
                    conflicts.append({"time": t, "type": "vertex", "i": i, "j": j,
                                      "cell": pi, "edge_from": None, "edge_to": None})
                if t < makespan:
                    ai_next = position_at(paths[i], t + 1)
                    aj_next = position_at(paths[j], t + 1)
                    if pi == aj_next and ai_next == pj and pi != ai_next:
                        pair_frequency[(i, j)] += 1
                        conflicts.append({"time": t, "type": "edge", "i": i, "j": j,
                                          "cell": None, "edge_from": pi, "edge_to": ai_next})
    for conflict in conflicts:
        pair_count = pair_frequency[(conflict["i"], conflict["j"])]
        base = 10.0 if conflict["type"] == "vertex" else 6.0
        site = conflict["cell"] if conflict["type"] == "vertex" else conflict["edge_to"]
        persistence = min(max(pair_count - 1, 0), 5) * 0.75
        conflict["severity"] = (base + 4.0 * local_obstacle_density(grid, site)
                                 + persistence + 0.01 * (makespan - conflict["time"]))
    return conflicts


def find_adaptive_conflict(grid, paths):
    conflicts = _all_conflicts(grid, paths)
    if not conflicts:
        return None
    # Resolve higher-severity conflicts first; for equal severity, earlier time wins.
    conflicts.sort(key=lambda c: (c["severity"], -c["time"], -c["i"], -c["j"]), reverse=True)
    return conflicts[0]


def count_conflicts(grid, paths):
    return detect_collisions(paths)["collision_events"]


def has_future_goal_constraint(constraints, goal, arrival_time, horizon):
    return any((t, goal) in constraints.vertex for t in range(arrival_time + 1, horizon + 1))


def respects_constraints(constraints, current, nxt, next_time):
    if (next_time, nxt) in constraints.vertex:
        return False
    if (next_time - 1, current, nxt) in constraints.edge:
        return False
    return True


def reconstruct_space_time_path(parent, goal_state):
    states = []
    current = goal_state
    while current is not None:
        states.append(current)
        current = parent[current]
    states.reverse()
    return [state[:2] for state in states]


def low_level_constrained_astar(grid, start, goal, constraints, horizon):
    """A* over (row, column, time); a WAIT action is allowed."""
    rows, cols = len(grid), len(grid[0])
    start_state = (start[0], start[1], 0)
    open_heap = []
    counter = 0
    g_score = {start_state: 0}
    parent = {start_state: None}
    closed = set()
    heappush(open_heap, (h_manhattan(start, goal), 0, counter, start_state))
    nodes_expanded = 0

    while open_heap:
        _, _, _, state = heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)
        nodes_expanded += 1
        r, c, t = state
        current = (r, c)

        if current == goal and not has_future_goal_constraint(constraints, goal, t, horizon):
            return reconstruct_space_time_path(parent, state), nodes_expanded
        if t >= horizon:
            continue

        for dr, dc in SPACE_TIME_MOVES:
            nr, nc = r + dr, c + dc
            nt = t + 1
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc]:
                continue
            nxt = (nr, nc)
            if not respects_constraints(constraints, current, nxt, nt):
                continue
            next_state = (nr, nc, nt)
            tentative_g = g_score[state] + 1
            if tentative_g < g_score.get(next_state, inf):
                g_score[next_state] = tentative_g
                parent[next_state] = state
                counter += 1
                heappush(open_heap, (tentative_g + h_manhattan(nxt, goal),
                                     -tentative_g, counter, next_state))
    return None, nodes_expanded


def path_respects_constraints(path, constraints, horizon):
    for t, cell in enumerate(path):
        if (t, cell) in constraints.vertex:
            return False
        if t > 0 and (t - 1, path[t - 1], cell) in constraints.edge:
            return False
    return not has_future_goal_constraint(constraints, path[-1], len(path) - 1, horizon)


def failed_result(start_time, nodes, high_level_expansions):
    return {"success": False, "paths": None, "makespan": inf, "sum_of_costs": inf,
            "total_nodes_expanded": nodes,
            "planning_time_ms": (time.perf_counter() - start_time) * 1000,
            "collision_events": inf, "vertex_collisions": inf,
            "edge_swap_collisions": inf, "collision_pairs": inf,
            "wait_steps": inf, "high_level_expansions": high_level_expansions}


def cba_star(grid, agents, max_expansions=MAX_HIGH_LEVEL_EXPANSIONS):
    """Conflict-Based Adaptive A* (project-level adaptive CBS customization).

    High level: pick an adaptive-severity conflict and branch on either robot.
    Low level: replan that robot only, using A* in space-time and wait actions.
    Node priority is a practical conflict-aware score, not an optimality proof.
    """
    start_time = time.perf_counter()
    n_agents = len(agents)
    paths, total_nodes = [], 0
    for agent in agents:
        res = astar_single(grid, agent.start, agent.goal)
        total_nodes += res["nodes_expanded"]
        if not res["found"]:
            return failed_result(start_time, total_nodes, 0)
        paths.append(res["path"])

    root_conflicts = count_conflicts(grid, paths)
    if root_conflicts == 0:
        return {"success": True, "paths": paths,
                "makespan": max(len(p) - 1 for p in paths),
                "sum_of_costs": sum(len(p) - 1 for p in paths),
                "total_nodes_expanded": total_nodes,
                "planning_time_ms": (time.perf_counter() - start_time) * 1000,
                "wait_steps": 0, "high_level_expansions": 0,
                **detect_collisions(paths)}

    horizon = max(len(p) - 1 for p in paths) + max(20, len(grid) + len(grid[0])) + 5 * n_agents
    root_constraints = [AgentConstraints() for _ in range(n_agents)]
    root_cost = sum(len(p) - 1 for p in paths)
    node_counter = 0
    open_heap = []
    root_signature = tuple(c.signature() for c in root_constraints)
    seen_signatures = {root_signature}
    heappush(open_heap, (root_cost + 12 * root_conflicts, root_cost,
                         root_conflicts, node_counter, root_constraints, paths))
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, current_cost, _, _, constraints, current_paths = heappop(open_heap)
        expansions += 1
        conflict = find_adaptive_conflict(grid, current_paths)
        if conflict is None:
            collision_stats = detect_collisions(current_paths)
            return {"success": collision_stats["collision_events"] == 0,
                    "paths": current_paths,
                    "makespan": max(len(p) - 1 for p in current_paths),
                    "sum_of_costs": current_cost,
                    "total_nodes_expanded": total_nodes,
                    "planning_time_ms": (time.perf_counter() - start_time) * 1000,
                    "wait_steps": sum(1 for p in current_paths
                                       for a, b in zip(p, p[1:]) if a == b),
                    "high_level_expansions": expansions, **collision_stats}

        t, kind, i, j = conflict["time"], conflict["type"], conflict["i"], conflict["j"]
        for chosen_agent in (i, j):
            child_constraints = [con.copy() for con in constraints]
            if kind == "vertex":
                child_constraints[chosen_agent].vertex.add((t, conflict["cell"]))
            elif chosen_agent == i:
                child_constraints[chosen_agent].edge.add(
                    (t, conflict["edge_from"], conflict["edge_to"]))
            else:
                child_constraints[chosen_agent].edge.add(
                    (t, conflict["edge_to"], conflict["edge_from"]))

            signature = tuple(con.signature() for con in child_constraints)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            new_path, expanded = low_level_constrained_astar(
                grid, agents[chosen_agent].start, agents[chosen_agent].goal,
                child_constraints[chosen_agent], horizon)
            total_nodes += expanded
            if new_path is None or not path_respects_constraints(
                    new_path, child_constraints[chosen_agent], horizon):
                continue

            child_paths = list(current_paths)
            child_paths[chosen_agent] = new_path
            child_conflicts = count_conflicts(grid, child_paths)
            child_cost = sum(len(p) - 1 for p in child_paths)
            node_counter += 1
            priority = child_cost + 12 * child_conflicts
            heappush(open_heap, (priority, child_cost, child_conflicts, node_counter,
                                 child_constraints, child_paths))

    return failed_result(start_time, total_nodes, expansions)


# ------------------------------ Pygame UI -----------------------------------
def run_app():
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "Pygame is not installed. Run: python -m pip install pygame"
        ) from exc

    pygame.init()
    pygame.display.set_caption("Multi-Robot Pathfinding — Independent A* vs CBA*")

    # Launch in a regular window (not fullscreen), as in the earlier version.
    # The window is resizable; the logical UI canvas is scaled to fit it.
    display = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    display_w, display_h = display.get_size()
    screen = pygame.Surface((SCREEN_W, SCREEN_H)).convert()
    scale_x, scale_y = display_w / SCREEN_W, display_h / SCREEN_H
    clock = pygame.time.Clock()

    # Slightly larger fonts and stronger weights improve readability, especially
    # on the controls and status/metric text when shown on a fullscreen display.
    font = pygame.font.SysFont("segoeui", 18)
    small = pygame.font.SysFont("segoeui", 16)
    tiny = pygame.font.SysFont("segoeui", 13)
    button_font = pygame.font.SysFont("segoeui", 16, bold=True)
    title_font = pygame.font.SysFont("segoeui", 25, bold=True)

    # Restrained, Apple-inspired palette: soft canvas, white cards, blue accent.
    BG = (245, 245, 247)
    PANEL = (255, 255, 255)
    PANEL_ALT = (242, 242, 247)
    WHITE = (29, 29, 31)
    MUTED = (110, 110, 115)
    GRID_LINE = (211, 211, 216)
    FREE = (250, 250, 252)
    OBSTACLE = (55, 60, 68)
    GOAL_FILL = (232, 232, 237)
    GREEN = (40, 167, 69)
    BLUE = (0, 122, 255)
    ORANGE = (196, 119, 0)
    RED = (255, 59, 48)

    config = {
        "rows": DEFAULT_ROWS,
        "cols": DEFAULT_COLS,
        "robots": DEFAULT_ROBOTS,
        "obstacle_percent": DEFAULT_OBSTACLE_PERCENT,
        "method": "CBA* (proposed)",
    }
    grid, agents, result = None, [], None
    paths = None
    status = "Set parameters on the right, then click PLAN / REGENERATE."
    status_colour = MUTED
    sim_t = 0
    playing = False
    speed = 1.0
    show_paths = True
    last_tick = time.perf_counter()
    buttons = []
    rng = random.Random()

    def text(surface, value, x, y, colour=WHITE, use_font=None):
        surf = (use_font or font).render(str(value), True, colour)
        surface.blit(surf, (x, y))

    def present():
        """Scale the logical UI canvas to fit the current window size."""
        frame = pygame.transform.smoothscale(screen, (display_w, display_h))
        display.blit(frame, (0, 0))
        pygame.display.flip()

    def register_button(rect, key, label, enabled=True, selected=False):
        buttons.append((pygame.Rect(rect), key, label, enabled, selected))

    def plan_scenario():
        nonlocal grid, agents, result, paths, sim_t, playing, status, status_colour, last_tick
        status = "Generating grid and planning paths..."
        status_colour = ORANGE
        screen.fill(BG)
        text(screen, "Planning — please wait", 35, 35, ORANGE, title_font)
        present()
        try:
            grid, agents = generate_multi_robot_scenario(
                config["rows"], config["cols"], config["obstacle_percent"],
                config["robots"], rng)
            if config["method"] == "Independent A*":
                result = independent_astar(grid, agents)
            else:
                result = cba_star(grid, agents)
            paths = result.get("paths")
            sim_t = 0
            # Planning/regeneration only creates the paths. Playback remains
            # paused until the user explicitly presses Play / Resume.
            playing = False
            last_tick = time.perf_counter()
            if paths is None:
                status = ("CBA* did not find a collision-free plan within its search limit. "
                          "Try fewer robots, a larger grid, or fewer obstacles.")
                status_colour = RED
            elif result.get("collision_events", 0):
                status = ("Plan ready at t=0, but Independent A* paths contain collisions. "
                          "Press Play / Resume to inspect them.")
                status_colour = ORANGE
            else:
                status = ("Plan ready. All robots start at t=0. Press Play / Resume "
                          "when you are ready.")
                status_colour = GREEN
        except (RuntimeError, ValueError) as exc:
            grid, agents, result, paths = None, [], None, None
            sim_t, playing = 0, False
            status = str(exc)
            status_colour = RED

    def change_setting(key, delta):
        if key == "robots":
            config[key] = max(2, min(12, config[key] + delta))
        elif key == "cols":
            config[key] = max(10, min(48, config[key] + delta))
        elif key == "rows":
            config[key] = max(8, min(32, config[key] + delta))
        elif key == "obstacle_percent":
            config[key] = max(0, min(40, config[key] + delta))

    def handle_button(key):
        nonlocal playing, sim_t, show_paths, speed, status, status_colour, last_tick
        if key == "method":
            config["method"] = "Independent A*" if config["method"] == "CBA* (proposed)" else "CBA* (proposed)"
            status = "Algorithm changed. Click PLAN / REGENERATE to apply it."
            status_colour = ORANGE
        elif key.endswith("_minus") or key.endswith("_plus"):
            base_key = key.rsplit("_", 1)[0]
            delta = -1 if key.endswith("_minus") else 1
            if base_key in ("obstacle_percent",):
                delta *= 5
            change_setting(base_key, delta)
            status = "Settings changed — click PLAN / REGENERATE to apply them."
            status_colour = ORANGE
        elif key == "run":
            plan_scenario()
        elif key == "pause":
            if paths:
                playing = not playing
                last_tick = time.perf_counter()
        elif key == "step":
            if paths:
                sim_t = min(sim_t + 1, max(len(p) - 1 for p in paths))
                playing = False
        elif key == "step_back":
            if paths:
                sim_t = max(0, sim_t - 1)
                playing = False
        elif key == "reset":
            sim_t = 0
            playing = False
            last_tick = time.perf_counter()
        elif key == "show_paths":
            show_paths = not show_paths
        elif key == "speed_minus":
            speed = max(0.25, round(speed - 0.25, 2))
        elif key == "speed_plus":
            speed = min(4.0, round(speed + 0.25, 2))

    def cell_geometry():
        x, y, w, h = GRID_AREA
        # Keep rendering the last generated grid while edited settings are pending.
        rows = len(grid) if grid is not None else config["rows"]
        cols = len(grid[0]) if grid is not None else config["cols"]
        cell_size = max(1, int(min(w / cols, h / rows)))
        grid_w, grid_h = cell_size * cols, cell_size * rows
        gx = x + (w - grid_w) // 2
        gy = y + (h - grid_h) // 2
        return gx, gy, cell_size, grid_w, grid_h

    def cell_center(cell, geom):
        gx, gy, size, _, _ = geom
        r, c = cell
        return gx + c * size + size // 2, gy + r * size + size // 2

    def draw_grid():
        if grid is None:
            pygame.draw.rect(screen, PANEL, pygame.Rect(GRID_AREA), border_radius=12)
            text(screen, "The grid will appear here", 300, 390, MUTED, title_font)
            return
        geom = cell_geometry()
        gx, gy, size, grid_w, grid_h = geom
        pygame.draw.rect(screen, FREE, (gx, gy, grid_w, grid_h))

        # Planned paths are drawn first, below cell borders, goals and robots.
        if show_paths and paths:
            path_surface = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
            for i, path in enumerate(paths):
                colour = (*AGENT_COLOURS[i % len(AGENT_COLOURS)], 115)
                points = [(c * size + size // 2, r * size + size // 2) for r, c in path]
                if len(points) >= 2:
                    pygame.draw.lines(path_surface, colour, False, points, max(2, size // 6))
                for px, py in points:
                    pygame.draw.circle(path_surface, colour, (px, py), max(2, size // 8))
            screen.blit(path_surface, (gx, gy))

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                rect = pygame.Rect(gx + c * size, gy + r * size, size, size)
                if grid[r][c]:
                    pygame.draw.rect(screen, OBSTACLE, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

        # Goals (one outline per robot).
        if agents:
            for i, agent in enumerate(agents):
                colour = AGENT_COLOURS[i % len(AGENT_COLOURS)]
                rr, cc = agent.goal
                inset = max(2, size // 6)
                goal_rect = pygame.Rect(gx + cc * size + inset, gy + rr * size + inset,
                                        max(2, size - 2 * inset), max(2, size - 2 * inset))
                pygame.draw.rect(screen, colour, goal_rect, max(2, size // 8), border_radius=3)
                if size >= 19:
                    tag = tiny.render(f"G{i+1}", True, (15, 20, 28))
                    screen.blit(tag, (gx + cc * size + 2, gy + rr * size + 2))

        # Show all start cells; their position is path[i][0] at shared t=0.
        if agents:
            for i, agent in enumerate(agents):
                colour = AGENT_COLOURS[i % len(AGENT_COLOURS)]
                sx, sy = cell_center(agent.start, geom)
                pygame.draw.circle(screen, colour, (sx, sy), max(3, size // 3), 2)
                if size >= 22:
                    tag = tiny.render(f"S{i+1}", True, colour)
                    screen.blit(tag, (gx + agent.start[1] * size + 1,
                                      gy + agent.start[0] * size + 1))

        if paths:
            # Mark current vertex conflicts in red.
            occupied = defaultdict(list)
            for i, path in enumerate(paths):
                p = position_at(path, sim_t)
                occupied[p].append(i)
            for cell, ids in occupied.items():
                if len(ids) > 1:
                    cx, cy = cell_center(cell, geom)
                    pygame.draw.circle(screen, RED, (cx, cy), max(4, size // 2 - 1), 3)
            # Mark edge swaps happening on the transition into the current tick.
            if sim_t > 0:
                for i in range(len(paths)):
                    a0, a1 = position_at(paths[i], sim_t - 1), position_at(paths[i], sim_t)
                    for j in range(i + 1, len(paths)):
                        b0, b1 = position_at(paths[j], sim_t - 1), position_at(paths[j], sim_t)
                        if a0 == b1 and a1 == b0 and a0 != a1:
                            p0, p1 = cell_center(a0, geom), cell_center(a1, geom)
                            pygame.draw.line(screen, RED, p0, p1, max(2, size // 5))

            # Draw robots on top.
            for i, path in enumerate(paths):
                cell = position_at(path, sim_t)
                cx, cy = cell_center(cell, geom)
                colour = AGENT_COLOURS[i % len(AGENT_COLOURS)]
                radius = max(4, size // 3)
                pygame.draw.circle(screen, (20, 25, 35), (cx + 1, cy + 2), radius + 1)
                pygame.draw.circle(screen, colour, (cx, cy), radius)
                pygame.draw.circle(screen, WHITE, (cx, cy), radius, 1)
                if size >= 22:
                    label = tiny.render(str(i + 1), True, (255, 255, 255))
                    screen.blit(label, label.get_rect(center=(cx, cy)))

        pygame.draw.rect(screen, (72, 86, 103), (gx, gy, grid_w, grid_h), 2)
        shown_cols = len(grid[0]) if grid is not None else config["cols"]
        shown_rows = len(grid) if grid is not None else config["rows"]
        text(screen, f"Displayed grid: X={shown_cols} columns  x  Y={shown_rows} rows",
             28, 82, MUTED, small)

    def draw_panel():
        nonlocal buttons
        buttons = []
        panel_rect = pygame.Rect(PANEL_X, 94, SCREEN_W - PANEL_X - 24, 782)
        pygame.draw.rect(screen, (225, 225, 230), panel_rect.move(0, 3), border_radius=22)
        pygame.draw.rect(screen, PANEL, panel_rect, border_radius=22)
        text(screen, "Simulation controls", PANEL_X + 22, 112, WHITE, title_font)
        text(screen, "Configure · plan · inspect", PANEL_X + 22, 143, MUTED, small)

        method_rect = (PANEL_X + 18, 168, 375, 38)
        register_button(method_rect, "method", f"Algorithm: {config['method']}", selected=True)

        labels = [
            ("robots", "Number of robots", f"{config['robots']}", 224),
            ("cols", "Grid width · columns", f"{config['cols']}", 279),
            ("rows", "Grid height · rows", f"{config['rows']}", 334),
            ("obstacle_percent", "Obstacle density", f"{config['obstacle_percent']}%", 389),
        ]
        for key, label, value, y in labels:
            text(screen, label, PANEL_X + 22, y, WHITE, small)
            text(screen, value, PANEL_X + 235, y + 28, GREEN, font)
            delta = 5 if key == "obstacle_percent" else 1
            register_button((PANEL_X + 180, y + 23, 38, 32), f"{key}_minus", "−")
            register_button((PANEL_X + 328, y + 23, 38, 32), f"{key}_plus", "+")

        register_button((PANEL_X + 18, 454, 375, 43), "run", "Plan / regenerate scenario", selected=True)
        register_button((PANEL_X + 18, 509, 375, 38), "pause",
                        "Pause" if playing else "Play / resume", enabled=bool(paths))
        register_button((PANEL_X + 18, 555, 180, 36), "step_back", "Step −1", enabled=bool(paths) and sim_t > 0)
        register_button((PANEL_X + 213, 555, 180, 36), "step", "Step +1", enabled=bool(paths))
        register_button((PANEL_X + 18, 599, 180, 36), "reset", "Reset to t = 0", enabled=bool(paths))
        register_button((PANEL_X + 213, 599, 180, 36), "show_paths",
                        f"Paths {'on' if show_paths else 'off'}", selected=show_paths)
        text(screen, f"Playback speed   {speed:.2f}×", PANEL_X + 22, 649, WHITE, small)
        register_button((PANEL_X + 270, 642, 50, 32), "speed_minus", "−")
        register_button((PANEL_X + 330, 642, 50, 32), "speed_plus", "+")

        text(screen, "STATUS", PANEL_X + 22, 688, MUTED, small)
        status_lines = wrap_text(status, 48)
        for idx, line in enumerate(status_lines[:2]):
            text(screen, line, PANEL_X + 22, 706 + idx * 15, status_colour, tiny)

        text(screen, "LIVE METRICS", PANEL_X + 22, 741, MUTED, small)
        if result is not None:
            metrics = [
                f"Time step: {sim_t}",
                f"Makespan: {result.get('makespan', '—')}",
                f"Collisions in plan: {result.get('collision_events', '—')}",
                f"Sum of path costs: {result.get('sum_of_costs', '—')}",
                f"Planning time: {result.get('planning_time_ms', 0):.1f} ms",
                f"Nodes expanded: {result.get('total_nodes_expanded', 0)}",
                f"High-level expansions: {result.get('high_level_expansions', 0)}",
            ]
        else:
            metrics = [f"Time step: {sim_t}", "Plan not generated yet."]
        for idx, line in enumerate(metrics):
            if 751 + idx * 17 < 870:
                text(screen, line, PANEL_X + 22, 758 + idx * 16, WHITE if idx < 4 else MUTED, tiny)

        # Buttons use calm fills, subtle borders and consistent rounded corners.
        raw_mouse_x, raw_mouse_y = pygame.mouse.get_pos()
        mouse_pos = (int(raw_mouse_x / scale_x), int(raw_mouse_y / scale_y))
        for rect, key, label, enabled, selected in buttons:
            hovered = rect.collidepoint(mouse_pos) and enabled
            if selected:
                base = BLUE
                label_colour = (255, 255, 255)
            elif not enabled:
                base = (239, 240, 244)
                label_colour = (105, 108, 116)
            else:
                base = (224, 226, 232) if hovered else PANEL_ALT
                label_colour = (29, 29, 31)
            pygame.draw.rect(screen, base, rect, border_radius=10)
            pygame.draw.rect(screen, (198, 201, 209), rect, 1, border_radius=10)
            label_surf = button_font.render(label, True, label_colour)
            screen.blit(label_surf, label_surf.get_rect(center=rect.center))

    def wrap_text(value, max_chars):
        words = value.split()
        lines, line = [], ""
        for word in words:
            if len(line) + len(word) + (1 if line else 0) > max_chars:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            lines.append(line)
        return lines

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Keep the application windowed while allowing the user to resize it.
                display = pygame.display.set_mode((max(900, event.w), max(650, event.h)), pygame.RESIZABLE)
                display_w, display_h = display.get_size()
                scale_x, scale_y = display_w / SCREEN_W, display_h / SCREEN_H
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and paths:
                    playing = not playing
                    last_tick = now
                elif event.key == pygame.K_RIGHT and paths:
                    sim_t = min(sim_t + 1, max(len(p) - 1 for p in paths))
                    playing = False
                elif event.key == pygame.K_LEFT and paths:
                    sim_t = max(0, sim_t - 1)
                    playing = False
                elif event.key == pygame.K_HOME and paths:
                    sim_t = 0
                    playing = False
                elif event.key == pygame.K_p:
                    show_paths = not show_paths
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                logical_pos = (int(event.pos[0] / scale_x), int(event.pos[1] / scale_y))
                for rect, key, label, enabled, selected in buttons:
                    if rect.collidepoint(logical_pos) and enabled:
                        handle_button(key)
                        break

        if playing and paths:
            makespan = max(len(p) - 1 for p in paths)
            interval = 0.34 / speed
            if now - last_tick >= interval:
                sim_t = min(sim_t + 1, makespan)
                last_tick = now
                if sim_t >= makespan:
                    playing = False

        screen.fill(BG)
        text(screen, "Multi-robot pathfinding", 28, 26, WHITE, title_font)
        text(screen, "Synchronized starts at t = 0     Independent A*  /  Adaptive CBA*", 28, 59, MUTED, small)
        draw_grid()
        draw_panel()
        text(screen, "SPACE  Play / pause     ← / →  Step backward / forward     HOME  Reset     P  Paths     ESC  Quit",
             28, SCREEN_H - 18, MUTED, tiny)
        present()

    pygame.quit()


def main():
    run_app()


if __name__ == "__main__":
    main()
