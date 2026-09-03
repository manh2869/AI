import numpy as np

# X = np.random.uniform(0, 10, 10)
# print(X.mean())
# print(X.var())
# print(X.squeeze())
# print(X.shape)
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
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


# mse = mean_squared_error(y, y_hat)

# r2 = r2_score(y, y_hat)
# print(mse)
# print(r2)
# result = (y - y_hat) ** 2
# print(result.mean())


#                               scaler

from sklearn.preprocessing import StandardScaler


# print(y.reshape(-1, 1)) must reshape befor scaler can't scaler [  15.43422492  -24.52677522  142.00003238  132.83489946   12.13510206]
# after reshape [[  15.43422492]
#                 [ -24.52677522]
#                 [ 142.00003238]
#                 [ 132.83489946]
#                 [  12.13510206]]


# scaler = StandardScaler()
# y_scaler = scaler.fit_transform(y.reshape(-1, 1))
# print(y)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2,random_state=42)
# print(y_scaler)
