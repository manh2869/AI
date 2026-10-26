import random

GRID = 3


def grid():
    return [[0 for _ in range(GRID)] for _ in range(GRID)]


rng = random.Random(11)
TERMINAL = (GRID - 1, GRID - 1)
ACTIONS = {
    "up": (-1, 0, 0.25),
    "down": (1, 0, 0.25),
    "left": (0, -1, 0.25),
    "right": (0, 1, 0.25),
}


def step(i, j, action):
    if (i, j) == TERMINAL:
        return i, j, 0.0, True
    dr, dc, _ = ACTIONS[action]

    nr = min(max(i + dr, 0), GRID - 1)
    nc = min(max(j + dc, 0), GRID - 1)
    return nr, nc, -1.0, (nr, nc) == TERMINAL


def random_action(actions, rng):
    x = rng.random()
    total = 0.0
    for a, p in actions.items():
        total += p[2]
        if x <= total:
            return a, p


def policy_evaluation(gamma=0.99, tol=1e-6, max_iter=2000):
    temp = grid()
    new_temp = [row[:] for row in temp]
    for _ in range(max_iter):
        for i in range(len(temp)):
            for j in range(len(temp[0])):
                if (i, j) == TERMINAL:
                    new_temp[i][j] = 0.0
                    continue
                Vs = 0
                for a in ACTIONS:
                    nr, nc, r, b = step(i, j, a)
                    Vs += 0.25 * (r + gamma * temp[nr][nc])
                new_temp[i][j] = Vs
        temp = new_temp
    return temp


print(policy_evaluation(max_iter=2000))


# def rollout(rng, max_steps=200):
#     state = [0, 0]
#     total = 0.0
#     steps = 0
#     for _ in range(max_steps):
#         a, p = random_action(ACTIONS, rng)
#         state, reward, done = step(state[0], state[1], a)
#         total += reward
#         steps += 1
#         if done:
#             break
#     return total, steps


# print(rollout(rng, max_steps=200))
# results = []
# step_counts = []

# for _ in range(1000):
#     total, steps = rollout(rng, max_steps=200)

#     results.append(total)
#     step_counts.append(steps)

# mean_reward = sum(results) / len(results)
# mean_steps = sum(step_counts) / len(step_counts)

# print("Mean reward:", mean_reward)
# print("Mean steps:", mean_steps)
