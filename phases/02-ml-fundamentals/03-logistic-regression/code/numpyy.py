import numpy as np

# X = np.random.uniform(0, 10, 10)
# print(X.mean())
# print(X.var())
# print(X.squeeze())
# print(X.shape)
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=10, n_features=2, noise=5, random_state=42)
# print(X)
# print(y)
model = LinearRegression()
model.fit(X, y)
y_hat = model.predict(X)

# print(X)
# print(y)

# print(model.coef_)
# print(model.intercept_)
# print(y_hat)


mes=mean

result = (y - y_hat) ** 2
print(result.mean())
