import random

TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 10

X = [random.randint(0, 10) for _ in range(N_SAMPLES)]
Y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X]

# from sklearn.datasets import make_regression
# X, Y = make_regression(n_samples=10, n_features=1, noise=2.0,random_state=42)
print(X)
print(Y)

class linearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, Y):
        n = len(Y)
        predictions = self.predict(X)
        cost = (
            sum((y_actual - y_pre) ** 2 for y_actual, y_pre in zip(Y, predictions)) / n
        )
        return cost
    
    def compute_gradients(self, X, Y):
        predictions = self.predict(X)
        n = len(Y)
        dw = (2 / n) * sum(
            ((pre - actual) * x for pre, actual, x in zip(predictions, Y, X))
        )
        db = (2 / n) * sum(((pre - actual) for pre, actual in zip(predictions, Y)))
        return dw, db

    def fit(self, X, Y, epoch=1000, print_every=200):
        for e in range(epoch):
            dw, db = self.compute_gradients(X, Y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, Y)
            self.cost_history.append(cost)
            if e % print_every == 0:
                print(
                    f"  Epoch {e:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}"
                )
        return self

model = linearRegression(learning_rate=0.001)
model.fit(X, Y, epoch=1000, print_every=10)
