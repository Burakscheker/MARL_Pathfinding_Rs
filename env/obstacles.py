"""Deterministik statik engel episode'ları."""
from dataclasses import dataclass

import numpy as np

from baselines.bfs_oracle import (
    bfs_dist,
    bfs_path,
    forbidden_from,
    manhattan,
)
# Bizdeki ad bfs_distance_map; kaynak dosya bfs_distances bekliyor.
from baselines.bfs_oracle import bfs_distance_map as bfs_distances

Cell = tuple[int, int]
Path = tuple[Cell, ...]
DIFFICULTIES = ("basic", "medium", "hard")
BASIC_SUBTYPES = ("single", "cluster", "short_wall", "edge_or_goal")
MEDIUM_SUBTYPES = (
    "horizontal_wall",
    "vertical_wall",
    "l_or_u",
    "two_detours",
)
HARD_SUBTYPES = (
    "dfs_corridor",
    "dead_end",
    "false_shortcut",
    "trace_alternative",
)


@dataclass(frozen=True)
class ObstacleConfig:
    start1: Cell
    start2: Cell
    goal: Cell
    obstacles: frozenset[Cell]
    difficulty: str
    subtype: str
    map_seed: int


def validate_obstacle_config(
        config: ObstacleConfig, n: int, max_steps: int,
        ) -> tuple[Path, Path]:
    """Aday haritanın kooperatif olarak çözülebilir olduğunu doğrula."""
    if config.difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty: {config.difficulty}")
    if {config.start1, config.start2, config.goal} & config.obstacles:
        raise ValueError("start and goal cells must be free")

    path1 = bfs_path(config.start1, config.goal, config.obstacles, n)
    if path1 is None or len(path1) - 1 > max_steps:
        raise ValueError("A1 has no valid static path within phase limit")

    blocked2 = config.obstacles | forbidden_from(
        path1, config.start1, config.start2, config.goal,
    )
    path2 = bfs_path(config.start2, config.goal, blocked2, n)
    if path2 is None or len(path2) - 1 > max_steps:
        raise ValueError("cooperative reference path is unavailable")
    return path1, path2


def _segment(start: Cell, end: Cell) -> set[Cell]:
    """Aynı satır veya sütundaki iki hücre arasını kapalı aralıkla döndür."""
    if start[0] == end[0]:
        lo, hi = sorted((start[1], end[1]))
        return {(start[0], col) for col in range(lo, hi + 1)}
    if start[1] == end[1]:
        lo, hi = sorted((start[0], end[0]))
        return {(row, start[1]) for row in range(lo, hi + 1)}
    raise ValueError("segment endpoints must share a row or column")


def _fill_to_target(
        obstacles: set[Cell],
        protected: set[Cell],
        target: int,
        n: int,
        rng: np.random.Generator,
        ) -> None:
    candidates = [
        (row, col)
        for row in range(n)
        for col in range(n)
        if (row, col) not in obstacles and (row, col) not in protected
    ]
    needed = target - len(obstacles)
    if needed < 0 or needed > len(candidates):
        raise ValueError("obstacle target is incompatible with protected paths")
    if needed:
        for index in rng.choice(len(candidates), size=needed, replace=False):
            obstacles.add(candidates[int(index)])


def _add_motif(
        obstacles: set[Cell],
        protected: set[Cell],
        subtype: str,
        n: int,
        rng: np.random.Generator,
        ) -> None:
    """Alt türün ayırt edici küçük şeklini güvenli bir alana ekle."""
    for _ in range(100):
        row = int(rng.integers(1, n - 4))
        col = int(rng.integers(1, n - 4))
        if subtype == "single":
            motif = {(row, col)}
        elif subtype == "cluster":
            motif = {
                (row + dr, col + dc)
                for dr in range(3) for dc in range(3)
            }
        elif subtype == "short_wall":
            motif = {(row, col + offset) for offset in range(4)}
        elif subtype == "edge_or_goal":
            motif = {(1, col), (1, col + 1), (2, col)}
        elif subtype == "horizontal_wall":
            motif = {(row, col + offset) for offset in range(n // 3)}
        elif subtype == "vertical_wall":
            motif = {(row + offset, col) for offset in range(n // 3)}
        elif subtype == "l_or_u":
            motif = (
                {(row + offset, col) for offset in range(10)}
                | {(row + 9, col + offset) for offset in range(8)}
                | {(row + offset, col + 7) for offset in range(6)}
            )
        elif subtype == "two_detours":
            motif = (
                {(row, col + offset) for offset in range(12)}
                | {(row + 8, col + offset) for offset in range(12)}
            )
        else:
            raise ValueError(f"unknown obstacle subtype: {subtype}")
        motif = {
            cell for cell in motif
            if 0 <= cell[0] < n and 0 <= cell[1] < n
        }
        if motif and not motif & protected:
            obstacles.update(motif)
            return
    raise ValueError(f"could not place obstacle subtype: {subtype}")


def _basic_candidate(
        map_seed: int, candidate_seed: int, n: int,
        subtype: str | None,
        ) -> ObstacleConfig:
    rng = np.random.default_rng(candidate_seed)
    selected = subtype or BASIC_SUBTYPES[
        int(rng.integers(0, len(BASIC_SUBTYPES)))
    ]
    if selected not in BASIC_SUBTYPES:
        raise ValueError(f"unknown basic subtype: {selected}")

    start1 = (0, int(rng.integers(0, n - 2)))
    start2 = (n - 1, int(rng.integers(0, n - 2)))
    goal = (int(rng.integers(1, n - 1)), n - 1)
    protected = (
        _segment(start1, (0, n - 1))
        | _segment((0, n - 1), goal)
        | _segment(start2, (n - 1, n - 1))
        | _segment((n - 1, n - 1), goal)
    )
    obstacles: set[Cell] = set()
    _add_motif(obstacles, protected, selected, n, rng)
    target = int(rng.integers(
        int(np.ceil(0.05 * n * n)),
        int(np.floor(0.10 * n * n)) + 1,
    ))
    _fill_to_target(obstacles, protected, target, n, rng)
    return ObstacleConfig(
        start1, start2, goal, frozenset(obstacles),
        "basic", selected, map_seed,
    )


def _medium_candidate(
        map_seed: int, candidate_seed: int, n: int,
        subtype: str | None,
        ) -> ObstacleConfig:
    rng = np.random.default_rng(candidate_seed)
    selected = subtype or MEDIUM_SUBTYPES[
        int(rng.integers(0, len(MEDIUM_SUBTYPES)))
    ]
    if selected not in MEDIUM_SUBTYPES:
        raise ValueError(f"unknown medium subtype: {selected}")

    wall_col = int(rng.integers(n // 3, 2 * n // 3))
    left_col = int(rng.integers(1, max(2, wall_col - 7)))
    right_col = int(rng.integers(
        min(n - 2, wall_col + 8), n - 1,
    ))
    width = right_col - left_col
    gap_offset = max(4, int(np.ceil(0.15 * width)))
    middle = int(rng.integers(
        gap_offset + 1, n - gap_offset - 1,
    ))
    upper_gap = middle - gap_offset
    lower_gap = middle + gap_offset
    start1 = (middle, left_col)
    start2 = (
        int(rng.integers(lower_gap + 1, n)),
        left_col,
    )
    goal = (middle, right_col)
    obstacles = {
        (row, wall_col)
        for row in range(n)
        if row not in {upper_gap, lower_gap}
    }
    path1 = (
        _segment(start1, (upper_gap, left_col))
        | _segment((upper_gap, left_col), (upper_gap, right_col))
        | _segment((upper_gap, right_col), goal)
    )
    path2 = (
        _segment(start2, (lower_gap, left_col))
        | _segment((lower_gap, left_col), (lower_gap, right_col))
        | _segment((lower_gap, right_col), goal)
    )
    protected = path1 | path2
    _add_motif(obstacles, protected | obstacles, selected, n, rng)
    target = int(rng.integers(
        int(np.ceil(0.12 * n * n)),
        int(np.floor(0.20 * n * n)) + 1,
    ))
    _fill_to_target(obstacles, protected, target, n, rng)
    return ObstacleConfig(
        start1, start2, goal, frozenset(obstacles),
        "medium", selected, map_seed,
    )


def _static_ratios(config: ObstacleConfig, n: int) -> tuple[float, float]:
    distances = (
        bfs_dist(config.start1, config.goal, config.obstacles, n),
        bfs_dist(config.start2, config.goal, config.obstacles, n),
    )
    if None in distances:
        return float("inf"), float("inf")
    return (
        distances[0] / manhattan(config.start1, config.goal),
        distances[1] / manhattan(config.start2, config.goal),
    )


def _randomized_dfs_free(n: int, rng: np.random.Generator) -> set[Cell]:
    """Üç hücre genişliğinde, DFS tabanlı bağlı bir koridor ağı kaz."""
    centers = list(range(2, n - 2, 4))
    nodes = [(row, col) for row in centers for col in centers]
    free: set[Cell] = set()
    for row, col in nodes:
        free.update({
            (row + dr, col + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        })

    start = nodes[int(rng.integers(0, len(nodes)))]
    visited = {start}
    stack = [start]
    while stack:
        row, col = stack[-1]
        neighbors = [
            (row + dr, col + dc)
            for dr, dc in ((-4, 0), (4, 0), (0, -4), (0, 4))
            if (row + dr, col + dc) in nodes
            and (row + dr, col + dc) not in visited
        ]
        if not neighbors:
            stack.pop()
            continue
        rng.shuffle(neighbors)
        next_row, next_col = neighbors[0]
        if row == next_row:
            gap_col = (col + next_col) // 2
            free.update({
                (row + offset, gap_col) for offset in (-1, 0, 1)
            })
        else:
            gap_row = (row + next_row) // 2
            free.update({
                (gap_row, col + offset) for offset in (-1, 0, 1)
            })
        visited.add((next_row, next_col))
        stack.append((next_row, next_col))
    return free


def _hard_candidate(
        map_seed: int,
        candidate_seed: int,
        n: int,
        max_steps: int,
        subtype: str | None,
        ) -> ObstacleConfig:
    rng = np.random.default_rng(candidate_seed)
    selected = subtype or HARD_SUBTYPES[
        int(rng.integers(0, len(HARD_SUBTYPES)))
    ]
    if selected not in HARD_SUBTYPES:
        raise ValueError(f"unknown hard subtype: {selected}")

    all_cells = {(row, col) for row in range(n) for col in range(n)}
    free = _randomized_dfs_free(n, rng)
    obstacles = all_cells - free
    target = int(rng.integers(
        int(np.ceil(0.22 * n * n)),
        int(np.floor(0.28 * n * n)) + 1,
    ))
    if len(obstacles) < target:
        raise ValueError("DFS corridor density is below hard range")

    # DFS ağındaki bazı duvarları kaldırmak döngü ve alternatif geçit üretir.
    removable = sorted(obstacles)
    remove_count = len(obstacles) - target
    for index in rng.choice(
            len(removable), size=remove_count, replace=False):
        obstacles.remove(removable[int(index)])

    free_cells = sorted(all_cells - obstacles)
    goal_order = rng.permutation(len(free_cells))
    for goal_index in goal_order[:80]:
        goal = free_cells[int(goal_index)]
        distances = bfs_distances(goal, obstacles, n)
        detour_starts = [
            cell for cell, distance in distances.items()
            if cell != goal
            and 4 <= manhattan(cell, goal)
            and distance <= max_steps
            and distance / manhattan(cell, goal) >= 1.50
        ]
        rng.shuffle(detour_starts)
        other_starts = [
            cell for cell, distance in distances.items()
            if cell != goal and distance <= max_steps
        ]
        rng.shuffle(other_starts)
        for start1 in detour_starts[:30]:
            for start2 in other_starts[:60]:
                if start2 == start1:
                    continue
                config = ObstacleConfig(
                    start1, start2, goal, frozenset(obstacles),
                    "hard", selected, map_seed,
                )
                try:
                    validate_obstacle_config(config, n, max_steps)
                except ValueError:
                    continue
                return config
    raise ValueError("DFS map has no cooperative hard start configuration")


def generate_obstacle_config(
        difficulty: str,
        map_seed: int,
        n: int = 50,
        max_steps: int = 150,
        subtype: str | None = None,
        ) -> ObstacleConfig:
    """İstenen zorlukta deterministik ve doğrulanmış bir episode üret."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty: {difficulty}")
    if n < 20:
        raise ValueError("obstacle generation requires n >= 20")

    for attempt in range(100):
        candidate_seed = map_seed * 100 + attempt
        if difficulty == "basic":
            config = _basic_candidate(map_seed, candidate_seed, n, subtype)
        elif difficulty == "medium":
            config = _medium_candidate(map_seed, candidate_seed, n, subtype)
        else:
            try:
                config = _hard_candidate(
                    map_seed, candidate_seed, n, max_steps, subtype,
                )
            except ValueError:
                continue
        try:
            validate_obstacle_config(config, n, max_steps)
        except ValueError:
            continue
        ratios = _static_ratios(config, n)
        if difficulty == "basic" and max(ratios) <= 1.15:
            return config
        if difficulty == "medium" and 1.20 <= max(ratios) <= 1.50:
            return config
        if difficulty == "hard" and max(ratios) >= 1.50:
            return config
    raise RuntimeError(
        f"could not generate {difficulty} obstacle map "
        f"for seed {map_seed} after 100 attempts"
    )
