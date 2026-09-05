import numpy as np
import math


y = ["Apple", "Apple", "Apple", "Orange", "Orange", "Orange", "Apple", "Orange"]

dic = {"Apple": 0, "Orange": 1}

y = [dic[label] for label in y]

x = np.array(
    [[150, 8], [160, 7], [140, 9], [300, 5], [320, 4], [280, 6], [155, 8], [310, 5]]
)
x_new = np.array([[200, 11]])


def euclidean_distance(x, x_new):
    result = []
    for i in x:
        result.append(math.sqrt(sum((sum((i - x_new) ** 2)))))
    return result


# help(np.sort)  helppppp

print(euclidean_distance(x, x_new))
