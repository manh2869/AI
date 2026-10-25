GRID = 5


def all_states():
    return [(r, c) for r in range(GRID) for c in range(GRID)]


TERMINAL = (4, 4)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def sample_action(dist, rng):
    x = rng.random()
    total = 0.0
    for a, p in dist.items():
        total += p
        if x <= total:
            return a
    return next(iter(dist))


def rollout(policy, rng, max_steps=200):
    state = (0, 0)
    total = 0.0
    steps = 0
    for _ in range(max_steps):
        action = sample_action(policy(state), rng)
        state, reward, done = step(state, action)
        total += reward
        steps += 1
        if done:
            break
    return total, steps


def policy_evaluation(policy, gamma=0.99, tol=1e-6, max_iter=2000):
    values = {s: 0.0 for s in all_states()}
    for _ in range(max_iter):
        delta = 0.0
        new_values = dict(values)
        for state in all_states():
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
