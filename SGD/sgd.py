#################################
# Guy Dankovich
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

# import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def get_prediction(w, x):
    max_w = 0
    max_mult = np.dot(w[0], x)
    for i in range(1, 10):
        val = np.dot(w[i], x)
        if val > max_mult:
            max_mult = val
            max_w = i
    return max_w


def get_p_distribution(y, x, w):
    dot_products = np.array([np.dot(w[i], x) for i in range(10)])
    overflow_products = np.exp(dot_products - np.max(dot_products))
    return overflow_products.item(y) / np.sum(overflow_products)


def accuracy_on_data_ce(data, labels, w):
    error_amount = 0
    for i in range(len(data)):
        if get_prediction(w, data[i]) != int(labels[i]):
            error_amount += 1
    return 1 - (error_amount / len(data))


def normalize_vector(v1):
    norm = np.linalg.norm(v1)
    if norm == 0:
        return v1
    return np.array([v1[i] / norm for i in range(len(v1))])


def dot_product_sign(v1, v2):
    if np.dot(v1, v2) >= 0:
        return 1
    return -1


def accuracy_on_data_hinge(data, labels, w):
    error_amount = 0
    for i in range(len(data)):
        if dot_product_sign(data[i], w) != labels[i]:
            error_amount += 1
    return 1 - (error_amount / len(data))


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    w = np.array([0 for i in range(784)])
    for i in range(1, T + 1):
        rand = numpy.random.randint(0, len(data) - 1)
        eta_t = eta_0 / i
        data_i = data[rand]
        if np.dot(labels[rand] * w, data_i) < 1:
            w = np.add((1 - eta_t) * w, eta_t * C * labels[rand] * data_i)
        else:
            w = np.multiply(1 - eta_t, w)
    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    w = [np.array([0 for i in range(784)]) for j in range(10)]
    for i in range(1, T + 1):
        rand = numpy.random.randint(0, len(data) - 1)
        data_i = data[rand]
        label = int(labels[rand])
        for j in range(10):
            distribution = get_p_distribution(j, data_i, w)
            if label == j:
                w[j] = np.subtract(w[j], (eta_0 * (distribution - 1) * data_i))
            else:
                w[j] = np.subtract(w[j], (eta_0 * distribution * data_i))
    return w


#################################
# code for section 1
# code for part (a)
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()

eta_0s = [10 ** i for i in range(-5, 6)]
accuracy_list = []
for eta_0 in eta_0s:
    accuracy_sum = 0
    for i in range(10):
        w = SGD_hinge(train_data, train_labels, 1, eta_0, 1000)
        accuracy_sum += accuracy_on_data_hinge(validation_data, validation_labels, w)
    accuracy_list.append(accuracy_sum / 10)

# plt.plot(eta_0s, accuracy_list)
# plt.xscale('log')
# plt.xlabel("eta 0's")
# plt.ylabel("accuracy")
# plt.show()

# code for part (b)
Cs = [10 ** i for i in range(-5, 6)]
accuracy_list = []
for c in Cs:
    accuracy_sum = 0
    for i in range(10):
        w = SGD_hinge(train_data, train_labels, c, 1, 1000)
        accuracy_sum += accuracy_on_data_hinge(validation_data, validation_labels, w)
    accuracy_list.append(accuracy_sum / 10)

# plt.plot(Cs, accuracy_list)
# plt.xscale('log')
# plt.xlabel("C values")
# plt.ylabel("accuracy")
# plt.show()

# code for part (c)
w = SGD_hinge(train_data, train_labels, 10 ** -4, 1, 20000)
# plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
# plt.show()

# code for part (d)
accuracy = accuracy_on_data_hinge(test_data, test_labels, w)
print("Accuracy of w for best eta and best C: ", accuracy)

# code for section 2
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()

# code for part (a)
eta_0s = [10 ** i for i in range(-10, 1)]
accuracy_list = []
for eta_0 in eta_0s:
    accuracy_sum = 0
    for i in range(10):
        w = SGD_ce(train_data, train_labels, eta_0, 1000)
        accuracy_sum += accuracy_on_data_ce(validation_data, validation_labels, w)
    accuracy_list.append(accuracy_sum / 10)

# plt.plot(eta_0s, accuracy_list)
# plt.xscale('log')
# plt.xlabel("eta_0")
# plt.ylabel("accuracy")
# plt.show()

# code for part (b)
w = SGD_ce(train_data, train_labels, 10 ** -6, 20000)
# for i in range(10):
#     plt.imshow(np.reshape(w[i], (28, 28)), interpolation='nearest')
#     plt.title("W" + str(i))
#     plt.show()

# code for part (c)
accuracy = accuracy_on_data_ce(test_data, test_labels, w)
print("Accuracy of w for best eta in CE: ", accuracy)

#################################
