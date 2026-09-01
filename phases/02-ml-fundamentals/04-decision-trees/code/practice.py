import math

# count = {}

# count = {"dog": 4, "cat": 1, "fish": 2}


# print(count.values())      dict_values([4, 1, 2])
# print(count.get("dog",2))  # if dog exists return value of dog else return 2
# print(count["cat"])       1
# count["manh"]=2 it is push


def gini_impurity(labels):  #  gini impurity of a node
    if len(labels) == 0:
        return 0.0
    count = {}
    for label in labels:
        count[label] = count.get(label, 0) + 1
    return 1 - sum((s / len(labels)) ** 2 for s in count.values())


def entropy(labels):
    n = len(labels)
    if len(labels) == 0:
        return 0.0
    count = {}
    for label in labels:
        count[label] = count.get(label, 0) + 1
    return -sum((s / n) * math.log2(s / n) for s in count.values() if s > 0)


# labels = [1, 1, 1, 0, 0]
# print(entropy(labels2))


def information_gain(parent, left, right, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent)
    n_l = len(left)
    n_r = len(right)
    if len(left) == 0 or len(right) == 0:
        return 0.0
    child_impurity = n_l / n * measure(left) + n_r / n * measure(right)
    return measure(parent) - child_impurity


labels2 = [1, 1, 1, 0, 0]

print(information_gain(labels2, [1, 1, 0], [0, 1], "entropy")) #very bad (a little bit of information)
print(information_gain(labels2, [1, 1, 0], [0, 1], "gini"))
