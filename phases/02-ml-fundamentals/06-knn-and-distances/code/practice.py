import numpy as np
import math


y = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange", "Apple", "Orange"]

# dic = {"Apple": 0, "Orange": 1}

# y = [dic[label] for label in y]

x = [[150, 8], [160, 7], [140, 9], [300, 5], [320, 4], [280, 6], [155, 8], [310, 5]]

x_new = [[200, 11]]


def euclidean_distance(x, x_new):
    result = []
    for i in x:
        result.append(math.sqrt((sum((i - x_new) ** 2))))
    return result


# help(np.sort)  helppppp

# print(euclidean_distance(x, x_new))

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x, y)
print(model.predict(x_new))
