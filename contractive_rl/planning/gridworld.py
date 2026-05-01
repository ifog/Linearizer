"""
Gridworld environment for rigorous planning experiments.

- Grid sizes: 10x10 (train) and 20x20 (test for generalization)
- Random obstacles (15% of cells blocked)
- Random goal
- Reward = -1/step, 0 at goal, gamma = 0.95
- compute_V_star via value iteration (tol=1e-6)
- bellman_backup: one step of Bellman
- generate_batch: (maps, V_stars)
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional

# Action deltas: right, left, down, up
ACTION_DELTAS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
GAMMA = 0.95
OBSTACLE_DENSITY = 0.15


def make_random_map(
    grid_size: int = 10,
    obstacle_density: float = OBSTACLE_DENSITY,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Generate a random gridworld map.

    Returns:
        obstacles: (grid_size, grid_size) bool array, True = obstacle
        goal: (row, col) tuple
    """
    rng = np.random.RandomState(seed)
    obstacles = rng.rand(grid_size, grid_size) < obstacle_density

    # Pick a random free goal cell
    while True:
        goal_r = rng.randint(0, grid_size)
        goal_c = rng.randint(0, grid_size)
        if not obstacles[goal_r, goal_c]:
            break

    obstacles[goal_r, goal_c] = False
    return obstacles, (goal_r, goal_c)


def _build_transition_table(
    obstacles: np.ndarray,
    goal: Tuple[int, int],
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute transition table for fast value iteration.

    Returns:
        next_states: (N, 4) int array, next state index per action
        is_goal: (N,) bool
        is_obstacle: (N,) bool
    """
    N = grid_size * grid_size
    goal_idx = goal[0] * grid_size + goal[1]
    obs_flat = obstacles.flatten()

    next_states = np.zeros((N, 4), dtype=np.int32)

    for s in range(N):
        r, c = divmod(s, grid_size)
        for a, (dr, dc) in enumerate(ACTION_DELTAS):
            nr, nc = r + dr, c + dc
            if (nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size
                    or obstacles[nr, nc]):
                nr, nc = r, c
            next_states[s, a] = nr * grid_size + nc

    is_goal = np.zeros(N, dtype=bool)
    is_goal[goal_idx] = True
    is_obstacle = obs_flat.copy()

    return next_states, is_goal, is_obstacle


def compute_V_star(
    grid: Tuple[np.ndarray, Tuple[int, int]],
    gamma: float = GAMMA,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> np.ndarray:
    """
    Compute V* via vectorized value iteration.

    Args:
        grid: (obstacles, goal) tuple as returned by make_random_map
        gamma: discount factor
        tol: convergence tolerance
        max_iter: maximum iterations

    Returns:
        V: (grid_size*grid_size,) float32 array, optimal value function (flattened)
    """
    obstacles, goal = grid
    grid_size = obstacles.shape[0]
    N = grid_size * grid_size
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)

    V = np.zeros(N, dtype=np.float32)
    free_mask = ~is_obstacle & ~is_goal

    for _ in range(max_iter):
        Q = -1.0 + gamma * V[next_states]  # (N, 4)
        V_new = Q.max(axis=1).astype(np.float32)
        V_new[is_goal] = 0.0
        V_new[is_obstacle] = 0.0

        delta = np.max(np.abs(V_new[free_mask] - V[free_mask])) if free_mask.any() else 0.0
        V = V_new
        if delta < tol:
            break

    return V


def bellman_backup(
    V: np.ndarray,
    grid: Tuple[np.ndarray, Tuple[int, int]],
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    One step of Bellman backup (vectorized).

    Args:
        V: (N,) float array
        grid: (obstacles, goal) tuple
        gamma: discount factor

    Returns:
        V_new: (N,) float32 array
    """
    obstacles, goal = grid
    grid_size = obstacles.shape[0]
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)

    Q = -1.0 + gamma * V[next_states]  # (N, 4)
    V_new = Q.max(axis=1).astype(np.float32)
    V_new[is_goal] = 0.0
    V_new[is_obstacle] = 0.0

    return V_new


def encode_map(
    obstacles: np.ndarray,
    goal: Tuple[int, int],
    grid_size: int,
) -> np.ndarray:
    """
    Encode obstacles + goal into a (3, H, W) float array.
    Channel 0: obstacle map
    Channel 1: goal map
    Channel 2: free space
    """
    obs_channel = obstacles.astype(np.float32)
    goal_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
    goal_channel[goal[0], goal[1]] = 1.0
    free_channel = (1.0 - obs_channel - goal_channel).clip(0.0, 1.0)
    return np.stack([obs_channel, goal_channel, free_channel], axis=0)  # (3, H, W)


def _generate_single_map(
    seed: int,
    grid_size: int,
    obstacle_density: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one map with V*, V_init and Bellman target."""
    obstacles, goal = make_random_map(
        grid_size=grid_size, obstacle_density=obstacle_density, seed=seed
    )
    grid = (obstacles, goal)
    V_star = compute_V_star(grid, gamma=gamma)
    map_enc = encode_map(obstacles, goal, grid_size=grid_size)

    rng = np.random.RandomState(seed + 12345)
    # Random initial V in [V_min, 0]
    v_min = V_star.min()
    V_init = rng.uniform(v_min, 0.0, size=(grid_size * grid_size,)).astype(np.float32)
    _, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)
    V_init[is_obstacle] = 0.0
    V_init[is_goal] = 0.0

    bellman_V = bellman_backup(V_init, grid, gamma=gamma)

    return map_enc, V_star, V_init, bellman_V


def generate_batch(
    n_maps: int,
    grid_size: int = 10,
    obstacle_density: float = OBSTACLE_DENSITY,
    gamma: float = GAMMA,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of (map_encoding, V_star, V_init, bellman_target).

    Returns:
        maps:      (n_maps, 3, H, W)  float32 tensor
        V_stars:   (n_maps, H*W)      float32 tensor
        V_inits:   (n_maps, H*W)      float32 tensor
        bellman_Vs:(n_maps, H*W)      float32 tensor
    """
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**28, size=n_maps)

    maps_list, V_stars_list, V_inits_list, bellman_list = [], [], [], []

    for s in seeds:
        map_enc, V_star, V_init, bellman_V = _generate_single_map(
            int(s), grid_size, obstacle_density, gamma
        )
        maps_list.append(map_enc)
        V_stars_list.append(V_star)
        V_inits_list.append(V_init)
        bellman_list.append(bellman_V)

    maps = torch.tensor(np.stack(maps_list, axis=0), dtype=torch.float32)
    V_stars = torch.tensor(np.stack(V_stars_list, axis=0), dtype=torch.float32)
    V_inits = torch.tensor(np.stack(V_inits_list, axis=0), dtype=torch.float32)
    bellman_Vs = torch.tensor(np.stack(bellman_list, axis=0), dtype=torch.float32)

    return maps, V_stars, V_inits, bellman_Vs


def get_greedy_policy(
    V_flat: np.ndarray,
    obstacles: np.ndarray,
    goal: Tuple[int, int],
    gamma: float = GAMMA,
) -> np.ndarray:
    """
    Derive greedy policy from value function V.

    Returns:
        policy: (grid_size, grid_size) array of action indices (0-3)
    """
    grid_size = obstacles.shape[0]
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)

    Q = -1.0 + gamma * V_flat[next_states]  # (N, 4)
    policy_flat = Q.argmax(axis=1)
    return policy_flat.reshape(grid_size, grid_size)


def simulate_episode(
    policy: np.ndarray,
    obstacles: np.ndarray,
    goal: Tuple[int, int],
    start: Optional[Tuple[int, int]] = None,
    max_steps: int = 200,
    seed: Optional[int] = None,
) -> Tuple[bool, int]:
    """
    Simulate an episode using the given policy.

    Returns:
        success: bool
        steps: int
    """
    grid_size = obstacles.shape[0]
    rng = np.random.RandomState(seed)
    goal_r, goal_c = goal

    if start is None:
        while True:
            r = rng.randint(0, grid_size)
            c = rng.randint(0, grid_size)
            if not obstacles[r, c] and not (r == goal_r and c == goal_c):
                break
    else:
        r, c = start

    for step in range(max_steps):
        if r == goal_r and c == goal_c:
            return True, step
        a = int(policy[r, c])
        dr, dc = ACTION_DELTAS[a]
        nr, nc = r + dr, c + dc
        if (nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size
                or obstacles[nr, nc]):
            nr, nc = r, c
        r, c = int(nr), int(nc)

    return (r == goal_r and c == goal_c), max_steps
