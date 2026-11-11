GRID = 4
TERMINAL = (3, 3)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
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
        "W1": [[rng.gauss(0, 0.2) for _ in range(n_in)] for _ in range(n_hidden)],
        "b1": [0.0] * n_hidden,
        "W2": [[rng.gauss(0, 0.2) for _ in range(n_hidden)] for _ in range(n_out)],
        "b2": [0.0] * n_out,
    }


import random

rng = random.Random()

net = {
    "W1": [[0.2, 0.5], [0.1, 0.3], [0.4, 0.7]],
    "b1": [0.1, 0.2, 0.1],
    "W2": [[0.2, 0.4, 0.1], [0.5, 0.3, 0.6]],
    "b2": [0.1, 0.2],
}

# net = init_net(2, 3, 2, rng)
# print(net)

x = [2, 3]


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


# print(forward(net, x))
# ([1.32, 3.3899999999999997], [2.0, 1.2999999999999998, 2.9999999999999996])


def clone(net):
    return {
        "W1": [row[:] for row in net["W1"]],
        "b1": net["b1"][:],
        "W2": [row[:] for row in net["W2"]],
        "b2": net["b2"][:],
    }


def train_step(online):
    n_hidden = len(online["b1"])
    n_out = len(online["b2"])
    n_in = len(online["W1"][0])
    print([[0.0] * n_in for _ in range(n_hidden)])
    print([0.0] * n_hidden)
    print([[0.0] * n_hidden for _ in range(n_out)])
    print([0.0] * n_out)
    total_loss = 0.0


# print(train_step(net))
# print(forward(net, x))
