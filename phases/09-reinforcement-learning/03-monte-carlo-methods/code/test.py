import random
from collections import defaultdict

GRID = 4
TERMINAL = (3, 3)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def reset():
    return (0, 0)


def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = DELTAS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def states():
    return [(r, c) for r in range(GRID) for c in range(GRID)]


def rollout(policy, rng, max_steps=200):
    trajectory = []
    state = reset()
    for _ in range(max_steps):
        action = policy(state, rng)
        state_next, reward, done = step(state, action)
        trajectory.append((state, action, reward))
        state = state_next
        if done:
            break
    return trajectory


def random_policy(state, rng):
    return rng.choice(ACTIONS)

rng = random.Random()
print(random_policy(1,rng))
