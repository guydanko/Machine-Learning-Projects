#################################
# Your name:Guy Dankovich
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
# from tabulate import tabulate
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
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


def dot_product_sign(v1, v2):
    if (np.dot(v1, v2) >= 0):
        return 1
    return -1


def normalize_vector(v1):
    norm = np.linalg.norm(v1)
    if norm == 0:
        return v1
    return np.array([v1[i] / norm for i in range(len(v1))])


def true_error_amount(test, labels, w):
    error_amount = 0
    for i in range(len(test)):
        if dot_product_sign(w, test[i]) != labels[i]:
            error_amount += 1
    return error_amount


def perceptron(data, labels):
    w = np.array([0 for i in range(784)])
    for i in range(len(data)):
        normalized_data = normalize_vector(data[i])
        if dot_product_sign(normalized_data, w) != labels[i]:
            w = np.add(labels[i] * data[i], w)
    return w
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """


# code for part (a)
n_vals = [5, 10, 50, 100, 500, 1000, 5000]
n_accuracy = []
five_percent = []
ninety_five_percent = []

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
train_vals = np.column_stack((train_data, train_labels))

for n in n_vals:
    mean_accuracy = []
    for i in range(100):
        errors = 0
        train_n = []
        labels_n = []
        np.random.shuffle(train_vals)
        for y in range(n):
            train_n.append(train_vals[y][:-1])
            labels_n.append(train_vals[y][-1])
        w = perceptron(train_n, labels_n)
        errors += true_error_amount(test_data, test_labels, w)
        mean_accuracy.append(1 - (errors / len(test_data)))

    mean_accuracy.sort()
    five_percent.append(mean_accuracy[5])
    ninety_five_percent.append(mean_accuracy[95])
    n_accuracy.append(np.mean(np.array(mean_accuracy)))

# headers = ["n", "Average Accuracy", "5th percentile accuracy", "95th percentile accuracy"]
# table = [[str(n_vals[i]), str(n_accuracy[i]), str(five_percent[i]), str(ninety_five_percent[i])] for i in
#          range(len(n_vals))]
# #print(tabulate(table, headers=headers, tablefmt='grid'))

# code for part (b)
w = perceptron(train_data, train_labels)
# plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
# plt.show()

# code for part (c)
error_amount = true_error_amount(test_data, test_labels, w)
accuracy = 1 - (error_amount / len(test_data))
print("Accuracy of perceptron on test data: ", accuracy)

# code for part (d)
found = 0
index_error = []
for i in range(len(test_data)):
    if found == 2:
        break
    if dot_product_sign(w, test_data[i]) != test_labels[i]:
        found += 1
        index_error.append(i)

# print("Perceptron gave label: ", dot_product_sign(w, test_data[index_error[0]]))
# plt.imshow(np.reshape(test_data[index_error[0]], (28, 28)), interpolation='nearest')
# plt.show()
#
# print("Perceptron gave label: ", dot_product_sign(w, test_data[index_error[1]]))
# plt.imshow(np.reshape(test_data[index_error[1]], (28, 28)), interpolation='nearest')
# plt.show()
