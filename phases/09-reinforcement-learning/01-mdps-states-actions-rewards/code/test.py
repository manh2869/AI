import random

GRID = 5


def grid():
    return [[_ for _ in range(GRID)] for _ in range(GRID)]


rng = random.Random()
TERMINAL = (GRID - 1, GRID - 1)
ACTIONS = {
    "up": (-1, 0, 0.25),
    "down": (1, 0, 0.25),
    "left": (0, -1, 0.25),
    "right": (0, 1, 0.25),
}


def step(i, j, action):
    if (i, j) == TERMINAL:
        return (i, j), 0.0, True
    dr, dc, _ = ACTIONS[action]

    nr = min(max(i + dr, 0), GRID - 1)
    nc = min(max(j + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def random_action(actions, rng):
    x = rng.random()
    total = 0.0
    for a, p in actions.items():
        total += p[2]
        if x <= total:
            return a, p


def rollout(rng, max_steps=200):
    state = [0, 0]
    total = 0.0
    steps = 0
    for _ in range(max_steps):
        a, p = random_action(ACTIONS, rng)
        state, reward, done = step(state[0], state[1], a)
        total += reward
        steps += 1
        if done:
            break
    return total, steps


print(rollout(rng, max_steps=200))
results = []
step_counts = []

for _ in range(1000):
    total, steps = rollout(rng, max_steps=200)

    results.append(total)
    step_counts.append(steps)

mean_reward = sum(results) / len(results)
mean_steps = sum(step_counts) / len(step_counts)

print("Mean reward:", mean_reward)
print("Mean steps:", mean_steps)


def policy_evaluation(policy, gamma=0.99, tol=1e-6, max_iter=2000):
    values = {s: 0.0 for s in grid()}
    for _ in range(max_iter):
        delta = 0.0
        new_values = dict(values)
        for state in grid():
            if state == TERMINAL:
                continue
            v = 0.0
            for action, pi_a in policy(state).items():
                s_next, reward, _ = step(state, action)
                v += pi_a * (reward + gamma * values[s_next])
            delta = max(delta, abs(v - values[state]))
            new_values[state] = v
        values = new_values
        if delta < tol:
            break
    return values
