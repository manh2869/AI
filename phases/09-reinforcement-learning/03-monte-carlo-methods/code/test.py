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


def rollout(max_steps=500):
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
    for i, j, a, r in reversed(trajectory):
        G = r + gamma * G
        cost.append([i, j, a, G])
    cost.reverse()
    return cost


def improve_policy():
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    for cost in episode_cost(rollout()):
        Q[(cost[0], cost[1])][cost[2]] += cost[3]
        counts[(cost[0], cost[1])][cost[2]] += 1




    def policy():
        pass
    
    return Q.items(), counts.items()


x, y = improve_policy()
for z in x:
    print(z)
for c in y:
    print(c)


# def average_episode():
#     m = grid()
#     count = grid()
#     for _ in range(600):
#         for x in episode_cost(rollout()):
#             m[x[0]][x[1]] += x[3]
#             count[x[0]][x[1]] += 1

#     ave = []
#     for i, j in zip(m, count):
#         temp = []
#         for x, y in zip(i, j):
#             if y == 0:
#                 temp.append(0)
#             else:
#                 temp.append(x / y)
#         ave.append(temp)
#     return ave

# for i in average_episode():
#     print(i)
