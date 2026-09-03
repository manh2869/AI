import numpy as np

# X = np.random.uniform(0, 10, 10)
# print(X.mean())
# print(X.var())
# print(X.squeeze())
# print(X.shape)
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=10, n_features=2, noise=20, random_state=42)
# print(X)
# print(y)
model = LinearRegression()
model.fit(X, y)
y_hat = model.predict(X)

print(X)
print(y)
print(model.coef_)
print(model.intercept_)
print(y_hat)
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Dữ liệu thật
ax.scatter(X[:, 0], X[:, 1], y, label="Actual")

# Tạo lưới để vẽ mặt phẳng dự đoán
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)

X1, X2 = np.meshgrid(x1, x2)

Y_hat = model.coef_[0] * X1 + model.coef_[1] * X2 + model.intercept_

# Mặt phẳng hồi quy
ax.plot_surface(X1, X2, Y_hat, alpha=0.5)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("y")

plt.savefig("regression.png")
