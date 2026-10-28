import random
from collections import defaultdict

GRID = 3
TERMINAL = (GRID - 1, GRID - 1)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
rng = random.Random(1)


def grid():
    return [[0 for _ in range(GRID)] for _ in range(GRID)]


def step(i, j, action):
    if (i, j) == TERMINAL:
        return i, j, 0.0, True
    dr, dc = DELTAS[action]
    r, c = i, j
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return nr, nc, -1.0, (nr, nc) == TERMINAL


def rollout(max_steps=200):
    trajectory = []
    i, j = 0, 0
    for _ in range(max_steps):
        action = rng.choice(ACTIONS)
        x, y, reward, done = step(i, j, action)
        trajectory.append([i, j, action, reward])
        i, j = x, y
        if done:
            break
    return trajectory


def episode_cost(trajectory, gamma=0.99):
    cost = []
    G = 0.0
    for i, j, _, r in reversed(trajectory):
        G = r + gamma * G
        cost.append([i, j, G])
    cost.reverse()
    return cost


def average_episode():
    m = grid()
    count = grid()

    for x in episode_cost(rollout()):
        m[x[0]][x[1]] += x[2]
        count[x[0]][x[1]] += 1

    return m


average_episode()


# print(rollout())
# for x in episode_cost(rollout()):
#     print(x[1][2])


# for m in rollout():
#     print(m)
# for m in episode_cost(rollout(), 0.99):
#     print(m)
