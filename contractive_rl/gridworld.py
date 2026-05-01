"""
Gridworld environment for Contractive Linearizer RL experiments.

- 10x10 grid
- Reward = -1 per step, 0 at goal
- Random obstacles and goal positions
- Ground-truth V* computed via vectorized value iteration (fast numpy)
"""

import numpy as np
import torch


GRID_SIZE = 10
GAMMA = 0.99

# Action deltas: right, left, down, up
ACTION_DELTAS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)


def make_random_map(grid_size=GRID_SIZE, obstacle_density=0.2, seed=None):
    """
    Generate a random gridworld map.

    Returns:
        obstacles: (grid_size, grid_size) bool array — True = obstacle
        goal: (row, col) tuple
    """
    rng = np.random.RandomState(seed)
    obstacles = rng.rand(grid_size, grid_size) < obstacle_density

    while True:
        goal_r = rng.randint(0, grid_size)
        goal_c = rng.randint(0, grid_size)
        if not obstacles[goal_r, goal_c]:
            break

    obstacles[goal_r, goal_c] = False
    return obstacles, (goal_r, goal_c)


def _build_transition_table(obstacles, goal, grid_size=GRID_SIZE):
    """
    Pre-compute transition table for fast value iteration.

    Returns:
        next_states: (grid_size*grid_size, 4) int array — next state index per action
        is_goal: (grid_size*grid_size,) bool
        is_obstacle: (grid_size*grid_size,) bool
    """
    N = grid_size * grid_size
    goal_idx = goal[0] * grid_size + goal[1]
    obs_flat = obstacles.flatten()

    next_states = np.zeros((N, 4), dtype=np.int32)

    for s in range(N):
        r, c = divmod(s, grid_size)
        for a, (dr, dc) in enumerate(ACTION_DELTAS):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size or obstacles[nr, nc]:
                nr, nc = r, c
            next_states[s, a] = nr * grid_size + nc

    is_goal = np.zeros(N, dtype=bool)
    is_goal[goal_idx] = True
    is_obstacle = obs_flat.copy()

    return next_states, is_goal, is_obstacle


def value_iteration(obstacles, goal, gamma=GAMMA, tol=1e-6, max_iter=500):
    """
    Compute V* via vectorized value iteration.

    Returns:
        V: (grid_size*grid_size,) float array — optimal value function (flattened)
    """
    grid_size = obstacles.shape[0]
    N = grid_size * grid_size
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)

    V = np.zeros(N, dtype=np.float32)
    free_mask = ~is_obstacle & ~is_goal  # states to update

    for _ in range(max_iter):
        # Q(s,a) = -1 + gamma * V[next_states]  for all s,a simultaneously
        Q = -1.0 + gamma * V[next_states]  # (N, 4)
        V_new = Q.max(axis=1).astype(np.float32)  # (N,)
        V_new[is_goal] = 0.0
        V_new[is_obstacle] = 0.0

        delta = np.max(np.abs(V_new[free_mask] - V[free_mask])) if free_mask.any() else 0.0
        V = V_new
        if delta < tol:
            break

    return V


def bellman_update(V_flat, obstacles, goal, gamma=GAMMA):
    """
    One step of Bellman backup (vectorized).

    Args:
        V_flat: (N,) float array

    Returns:
        V_new: (N,) float array
    """
    grid_size = obstacles.shape[0]
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)

    Q = -1.0 + gamma * V_flat[next_states]  # (N, 4)
    V_new = Q.max(axis=1).astype(np.float32)
    V_new[is_goal] = 0.0
    V_new[is_obstacle] = 0.0

    return V_new


def encode_map(obstacles, goal, grid_size=GRID_SIZE):
    """
    Encode obstacles + goal into a (3, H, W) float array.
    Channel 0: obstacle map, Channel 1: goal map, Channel 2: free space
    """
    obs_channel = obstacles.astype(np.float32)
    goal_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
    goal_channel[goal[0], goal[1]] = 1.0
    free_channel = (1.0 - obs_channel - goal_channel).clip(0, 1)
    return np.stack([obs_channel, goal_channel, free_channel], axis=0)  # (3, H, W)


# ---------------------------------------------------------------------------
# Pre-generate maps in parallel for fast batch creation
# ---------------------------------------------------------------------------

def _generate_single(args):
    """Helper for multiprocessing."""
    seed, grid_size, obstacle_density, gamma = args
    obstacles, goal = make_random_map(grid_size=grid_size, obstacle_density=obstacle_density, seed=seed)
    V_star = value_iteration(obstacles, goal, gamma=gamma)
    map_enc = encode_map(obstacles, goal, grid_size=grid_size)

    rng = np.random.RandomState(seed + 12345)
    V_init = rng.uniform(V_star.min(), 0.0, size=(grid_size * grid_size,)).astype(np.float32)
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)
    V_init[is_obstacle] = 0.0
    V_init[is_goal] = 0.0

    bellman_V = bellman_update(V_init, obstacles, goal, gamma=gamma)

    return map_enc, V_star, V_init, bellman_V


def generate_batch(batch_size, grid_size=GRID_SIZE, obstacle_density=0.2, gamma=GAMMA, seed=None):
    """
    Generate a batch of (map_encoding, V_star, V_init, bellman_target) tuples.

    Returns:
        maps: (B, 3, H, W) tensor
        V_star: (B, H*W) tensor
        V_init: (B, H*W) tensor
        bellman_V: (B, H*W) tensor
    """
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**28, size=batch_size)

    maps_list = []
    V_stars = []
    V_inits = []
    bellman_Vs = []

    for s in seeds:
        map_enc, V_star, V_init, bellman_V = _generate_single(
            (int(s), grid_size, obstacle_density, gamma)
        )
        maps_list.append(map_enc)
        V_stars.append(V_star)
        V_inits.append(V_init)
        bellman_Vs.append(bellman_V)

    maps = torch.tensor(np.stack(maps_list, axis=0), dtype=torch.float32)
    V_stars = torch.tensor(np.stack(V_stars, axis=0), dtype=torch.float32)
    V_inits = torch.tensor(np.stack(V_inits, axis=0), dtype=torch.float32)
    bellman_Vs = torch.tensor(np.stack(bellman_Vs, axis=0), dtype=torch.float32)

    return maps, V_stars, V_inits, bellman_Vs


def get_greedy_policy(V_flat, obstacles, goal, grid_size=GRID_SIZE, gamma=GAMMA):
    """
    Derive greedy policy from value function V.

    Returns:
        policy: (grid_size, grid_size) array of action indices (0-3)
    """
    N = grid_size * grid_size
    next_states, is_goal, is_obstacle = _build_transition_table(obstacles, goal, grid_size)

    Q = -1.0 + gamma * V_flat[next_states]  # (N, 4)
    policy_flat = Q.argmax(axis=1)
    return policy_flat.reshape(grid_size, grid_size)


def simulate_episode(policy, obstacles, goal, start=None, max_steps=200, grid_size=GRID_SIZE, seed=None):
    """
    Simulate an episode using the given policy.

    Returns:
        success: bool
        steps: int
    """
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
        a = policy[r, c]
        dr, dc = ACTION_DELTAS[a]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= grid_size or nc < 0 or nc >= grid_size or obstacles[nr, nc]:
            nr, nc = r, c
        r, c = int(nr), int(nc)

    return (r == goal_r and c == goal_c), max_steps
