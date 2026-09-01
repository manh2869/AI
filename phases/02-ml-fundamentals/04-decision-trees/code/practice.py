# count = {}

# count = {"dog": 4, "cat": 1, "fish": 2}


# print(count.values())      dict_values([4, 1, 2])
# print(count.get("dog",2))  # if dog exists return value of dog else return 2
# print(count["cat"])       1
# count["manh"]=2 it is push


def gini_impurity(labels):  #gini in a node of tree 
    if len(labels) == 0:
        return 0.0
    count = {}
    for label in labels:
        count[label] = count.get(label, 0) + 1
    return 1 - sum((s / len(labels)) ** 2 for s in count.values())


labels = [1, 1, 1, 0, 0]
print(gini_impurity(labels))
