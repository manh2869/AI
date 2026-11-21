import random

rng = random.Random(1)

GRID = 4
TERMINAL = (2, 2)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def step(state, action):
    if state == TERMINAL:
        return state, 10.0, True
    dr, dc = DELTAS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def state_features(state):
    feat = [0.0] * (GRID * GRID)
    r, c = state
    feat[r * GRID + c] = 1.0
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return feat


def init_net(n_in, n_hidden, n_out, rng):
    return {
        "W1": [
            [rng.randint(-20, 20) / 100 for _ in range(n_in)] for _ in range(n_hidden)
        ],
        "b1": [0.0] * n_hidden,
        "W2": [
            [rng.randint(-20, 20) / 100 for _ in range(n_hidden)] for _ in range(n_out)
        ],
        "b2": [0.0] * n_out,
    }


def forward(net, x):
    h = []
    for row, b in zip(net["W1"], net["b1"]):
        z = b + sum(w * xi for w, xi in zip(row, x))
        h.append(max(0.0, z))
    q = []
    for row, b in zip(net["W2"], net["b2"]):
        z = b + sum(w * hi for w, hi in zip(row, h))
        q.append(z)
    return q, h


def epsilon_greedy(net, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    q, _ = forward(net, state_features(state))
    return max(range(len(ACTIONS)), key=lambda i: q[i])


def clone(net):
    return {
        "W1": [row[:] for row in net["W1"]],
        "b1": net["b1"][:],
        "W2": [row[:] for row in net["W2"]],
        "b2": net["b2"][:],
    }


def train_step(online, target, batch, gamma, lr):
    n_hidden = len(online["b1"])
    n_out = len(online["b2"])
    n_in = len(online["W1"][0])
    dW1 = [[0.0] * n_in for _ in range(n_hidden)]
    db1 = [0.0] * n_hidden
    dW2 = [[0.0] * n_hidden for _ in range(n_out)]
    db2 = [0.0] * n_out
    total_loss = 0.0
    for s, a, r, s_next, done in batch:
        x = state_features(s)
        q, h = forward(online, x)
        if done:
            y = r
        else:
            q_next, _ = forward(target, state_features(s_next))
            y = r + gamma * max(q_next)

        td_error = q[a] - y
        total_loss += 0.5 * td_error * td_error

        db2[a] += td_error
        for j in range(n_hidden):
            dW2[a][j] += td_error * h[j]

        grad_h = [0.0] * n_hidden
        for j in range(n_hidden):
            if h[j] > 0:
                grad_h[j] = td_error * online["W2"][a][j]

        for j in range(n_hidden):
            db1[j] += grad_h[j]
            for k in range(n_in):
                dW1[j][k] += grad_h[j] * x[k]

    scale = lr / len(batch)

    for j in range(n_hidden):
        online["b1"][j] -= scale * db1[j]
        for k in range(n_in):
            online["W1"][j][k] -= scale * dW1[j][k]
    for a in range(n_out):
        online["b2"][a] -= scale * db2[a]
        for j in range(n_hidden):
            online["W2"][a][j] -= scale * dW2[a][j]

    return total_loss / len(batch)


online = init_net(16, 32, 4, rng)
target = clone(online)
return_log = []
buffer = []
step_count = 0

batch_size = 32

sync_every = 4

for ep in range(400):
    s = (0, 0)
    total = 0
    epsilon = max(0.05, 1 - ep / 200)
    for t in range(50):
        a = epsilon_greedy(online, s, rng, epsilon)

        s_next, r, done = step(s, ACTIONS[a])

        total += r

        buffer.append((s, a, r, s_next, done))

        if len(buffer) > 2000:
            buffer.pop(0)

        if len(buffer) >= batch_size:
            mb = rng.sample(buffer, batch_size)
            train_step(online, target, mb, gamma=0.99, lr=0.05)
        step_count += 1
        if step_count % sync_every == 0:
            target = clone(online)
        if done:
            break
        s = s_next
    return_log.append(total)

for i in range(GRID):
    temp = []
    for j in range(GRID):
        temp.append(ACTIONS[epsilon_greedy(online, [i, j], rng, 0)])
        pass
    print(temp)


# q, h = forward(online, state_features([2, 3]))

# ACTIONS = ("up", "down", "left", "right")

# print(epsilon_greedy(online, [2, 3], rng, 0.05))
