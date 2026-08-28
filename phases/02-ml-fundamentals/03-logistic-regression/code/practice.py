import random
import math

N = 200
x = []
y = []

random.seed(42)
                    # PERSISTENCE
                    # Success requires persistence.
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

# X = list(X) [[3.6671395552248383, 6.254212171523317], [5.007156919894976, 5.6252610870992195], [1.2461157367894518, 2.3890675109584354], [1.9265796588513269, 0.5622369445178927], [4.249436675221158, 6.678816131261848], [1.9128543873174506, 1.4434841520725123], [0.8256367783413008, 2.1966787467901563], [3.569577730566729, 4.520388533259548], [2.4267416277134544, 2.5551783572805316], [3.7669414415750886, 5.412843595836012]]
# Y = list(Y)  [1, 1, 0, 0, 1, 0, 0, 1, 0, 1]

print(X)
print(Y)


class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    # z=wx + b      with one feature
    # z=w1​x1​+w2​x2​+b with more feature
    # x = [1.85, 0.52]
    # weights = [0.7, -0.3]
    # bias = 0.5
    
    def predict_probability(self, X):
        z = sum(x1 * w1 for x1, w1 in zip(self.weights, X)) + self.bias
        return sigmoid(z)

    def classificasion(self, X, threshold=0.5):
        return 1 if self.predict_probability(X) >= threshold else 0

    #   −n1​i=1∑n​[yi​log(pi​)+(1−yi​)log(1−pi​)]
    def compute_loss(self, X, Y):
        n = len(Y)
        for i in range(n):
            p = self.predict_probability(X[i])
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self,X,Y,epochs=1000,times_print=200):
        n=len(y)
        n_features=X[0]
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db=0.0
            for i in range(n):
                p=self.predict_probability(X[i])
                erorr=p-Y[i]
                
                