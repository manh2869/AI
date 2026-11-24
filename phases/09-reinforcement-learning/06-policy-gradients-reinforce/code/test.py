import math
import random


GRID = 4
TERMINAL = (3, 3)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
N_ACTIONS = len(ACTIONS)
N_FEAT = GRID * GRID
rng = random.Random(42)


def reset():
    return (0, 0)


def init_theta(rng):
    return [[rng.gauss(0, 0.1) for _ in range(N_FEAT)] for _ in range(N_ACTIONS)]


def features(state):
    x = [0.0] * N_FEAT
    r, c = state
    x[r * GRID + c] = 1.0
    return x


def logits(theta, x):
    return [sum(w * xi for w, xi in zip(theta[a], x)) for a in range(N_ACTIONS)]


# print(features([1, 1])) [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# print(logits(init_theta(rng), features([1, 1])))
# [-0.14973534143409575, -0.1453545298008678, -0.12190747412956954, 0.2335754222480548]


def softmax(z):
    m = max(z)
    exps = [math.exp(zi - m) for zi in z]
    Z = sum(exps)
    return [e / Z for e in exps]


# print(softmax(logits(init_theta(rng), features([1, 1]))))  logits ->  probability [0.2514748604222024, 0.24385064475063273, 0.23662857663240944, 0.2680459181947554]
def sample(probs, rng):
    x = rng.random()
    cum = 0.0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return N_ACTIONS - 1


def step(state, action_idx):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = DELTAS[ACTIONS[action_idx]]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def rollout(theta, rng, max_steps=100):
    traj = []
    s = reset()
    for _ in range(max_steps):
        x = features(s)
        probs = softmax(logits(theta, x))
        a = sample(probs, rng)
        s_next, r, done = step(s, a)
        traj.append((x, a, r, probs))
        if done:
            break
        s = s_next
    return traj


def returns_to_go(traj, gamma):
    G = 0.0
    out = []
    for _, _, r, _ in reversed(traj):  #  end ->   begin
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


# print(rollout(init_theta(rng), rng))
# print(returns_to_go(rollout(init_theta(rng), rng), 0.9))


# theta = init_theta(rng)

# traj = rollout(theta, rng)
# returns = returns_to_go(traj, 0.99)
# print(traj)
# print(returns)


def reinforce(episodes, lr=0.05, gamma=0.99, use_baseline=False, rng=None):
    rng = rng or random.Random(0)
    theta = init_theta(rng)
    baseline = 0.0
    returns_log = []
    for ep in range(episodes):
        traj = rollout(theta, rng)
        returns = returns_to_go(traj, gamma)
        if use_baseline:
            baseline = 0.95 * baseline + 0.05 * returns[0]
        for (x, a, _r, probs), G in zip(traj, returns):
            adv = G - (baseline if use_baseline else 0.0)
            for i in range(N_ACTIONS):
                grad = (1.0 if i == a else 0.0) - probs[i]
                for j in range(N_FEAT):
                    theta[i][j] += lr * adv * grad * x[j]
        returns_log.append(returns[0] if returns else 0.0)
    return theta


