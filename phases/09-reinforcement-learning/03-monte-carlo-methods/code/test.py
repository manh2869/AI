import random
from collections import defaultdict

GRID = 4
TERMINAL = (GRID - 1, GRID - 1)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
rng = random.Random(11)


def reset():
    return (0, 0)


def step(i, j, action):
    if (i, j) == TERMINAL:
        return i, j, 0.0, True
    dr, dc = DELTAS[action]
    r, c = i, j
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return nr, nc, -1.0, (nr, nc) == TERMINAL


def grid():
    return [[0 for _ in range(GRID)] for _ in range(GRID)]


def rollout(rng, max_steps=200):
    trajectory = []
    i, j = reset()
    for _ in range(max_steps):
        action = rng.choice(ACTIONS)
        x, y, reward, done = step(i, j, action)
        trajectory.append([[i, j], action, reward])
        i, j = x, y
        if done:
            break
    return trajectory


def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def mc_policy_evaluation(episodes, policy, gamma=0.99, rng=None):
    rng = rng or random.Random(0)
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(rng)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for [[x, y], _, _], G in zip(trajectory, returns):
            if ([x, y]) in seen:
                continue
            seen.add([x, y])
            counts[[x, y]] += 1
            V[[x, y]] += (G - V[[x, y]]) / counts[[x, y]]
    return V, counts


for m in rollout(rng):
    print(m)
for m in returns_from(rollout(rng), 0.99):
    print(m)
print(mc_policy_evaluation(200, rng.choice(ACTIONS)))
