#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    svclassifier = svm.SVC(C=1000, kernel='linear')
    svclassifier.fit(X_train, y_train)
    linear_svm_amount = svclassifier.n_support_
    create_plot(X_train, y_train, svclassifier)
    # plt.title("Linear Kernel SVM\n Amount of support vectors: {}".format(linear_svm_amount[0]+linear_svm_amount[1]))
    # plt.xlabel("X-values")
    # plt.ylabel("Y-values")
    # plt.show()
    print(linear_svm_amount)
    svclassifier = svm.SVC(C=1000, kernel='poly', degree=2)
    svclassifier.fit(X_train, y_train)
    quad_svm_amount = svclassifier.n_support_
    print(quad_svm_amount)
    create_plot(X_train, y_train, svclassifier)
    # plt.title("Quadratic kernel SVM\n Amount of support vectors: {}".format(quad_svm_amount[0] + quad_svm_amount[1]))
    # plt.xlabel("X-values")
    # plt.ylabel("Y-values")
    # plt.show()
    svclassifier = svm.SVC(C=1000, kernel='rbf')
    svclassifier.fit(X_train, y_train)
    rbf_svm_amount = svclassifier.n_support_
    print(rbf_svm_amount)
    create_plot(X_train, y_train, svclassifier)
    # plt.title("RBF kernel SVM\n Amount of support vectors: {}".format(rbf_svm_amount[0] + rbf_svm_amount[1]))
    # plt.xlabel("X-values")
    # plt.ylabel("Y-values")
    # plt.show()
    return np.array([linear_svm_amount, quad_svm_amount, rbf_svm_amount])


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_values = [10 ** i for i in range(-5, 6)]
    c_index = 0
    test_accuracy = []
    train_accuracy = []
    for i in range(-5, 6):
        svclassifier = svm.SVC(C=C_values[c_index], kernel='linear')
        svclassifier.fit(X_train, y_train)
        create_plot(X_train, y_train, svclassifier)
        # plt.title("Linear kernel SVM with C penalty : 10^{}".format(i))
        # plt.xlabel("X-values")
        # plt.ylabel("Y-values")
        # plt.show()
        train_accuracy.append(svclassifier.score(X_train, y_train))
        test_accuracy.append(svclassifier.score(X_val, y_val))
        c_index += 1

    # plt.plot(C_values, train_accuracy, color='blue')
    # plt.plot(C_values, test_accuracy, color='red')
    # plt.xscale('log')
    # plt.xlabel("C values")
    # plt.ylabel("accuracy")
    # plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    # plt.show()


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamma_values = [10 ** i for i in range(-5, 6)]
    gamma_index = 0
    test_accuracy = []
    train_accuracy = []
    for i in range(-5, 6):
        svclassifier = svm.SVC(C=10, kernel='rbf', gamma=gamma_values[gamma_index])
        svclassifier.fit(X_train, y_train)
        create_plot(X_train, y_train, svclassifier)
        plt.title("RBF kernel SVM with gamma: 10^{}".format(i))
        plt.xlabel("X-values")
        plt.ylabel("Y-values")
        plt.show()
        train_accuracy.append(svclassifier.score(X_train, y_train))
        test_accuracy.append(svclassifier.score(X_val, y_val))
        gamma_index += 1

    plt.plot(gamma_values, train_accuracy, color='blue')
    plt.plot(gamma_values, test_accuracy, color='red')
    plt.xscale('log')
    plt.xlabel("gamma values")
    plt.ylabel("accuracy")
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower center')
    plt.show()


train_data, training_labels, validation_data, validation_labels = get_points()

# train_three_kernels(train_data, training_labels, validation_data, validation_labels)
# linear_accuracy_per_C(train_data, training_labels, validation_data, validation_labels)
rbf_accuracy_per_gamma(train_data, training_labels, validation_data, validation_labels)
