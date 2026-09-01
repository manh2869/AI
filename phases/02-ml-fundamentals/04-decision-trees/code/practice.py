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


labels = [1, 1, 1, 0, 0]
labels2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

print(gini_impurity(labels))
print(entropy(labels2))
