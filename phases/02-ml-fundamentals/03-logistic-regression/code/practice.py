import random
import math

N = 100
x = []
y = []


def sigmoid(z):
    z = max(-500, min(500, z))  #  z only in interval [-500,500]
    return 1 / (1 + math.e ** (-z))


for _ in range(N // 2):
    x.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)
for _ in range(N // 2):
    x.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combine = list(zip(x, y))
random.shuffle(combine)
X, Y = zip(*combine)
X = list(X)
Y = list(Y)


class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    # z=wx + b
    # z=w1​x1​+w2​x2​+b with more feature
    # x = [1.85, 0.52]
    # weights = [0.7, -0.3]
    # bias = 0.5
    def predict_probability(self, X):
        z = sum(x1 * w1 for x1, w1 in zip(self.weights, X)) + self.bias
        return sigmoid(z)
