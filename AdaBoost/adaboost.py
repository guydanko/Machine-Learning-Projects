#################################
# Your name: Guy Dankovich
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def get_weak_learner(D, X_train, y_train, lower_val):
    zipped_data = zip(X_train, y_train, D)
    best_error = 2  # bigger than 1
    dim_for_h = 0
    best_threshold = 0

    sorted_in_d = zipped_data

    for d in range(len(X_train[0])):
        sorted_in_d = sorted(sorted_in_d, key=lambda point: point[0][d])
        error = np.sum([point[2] for point in sorted_in_d if point[1] == lower_val])  # error if thresh is first point

        if error < best_error:
            best_threshold = sorted_in_d[0][0][d] - 1
            best_error = error
            dim_for_h = d

        # from first point to one before last
        for i in range(len(X_train) - 1):
            if sorted_in_d[i][1] == lower_val:
                error -= sorted_in_d[i][2]
            else:
                error += sorted_in_d[i][2]
            if error < best_error and sorted_in_d[i][0][d] != sorted_in_d[i + 1][0][d]:
                best_error = error
                best_threshold = (sorted_in_d[i][0][d] + sorted_in_d[i + 1][0][d]) / 2
                dim_for_h = d

        # last point
        if sorted_in_d[len(X_train) - 1][1] == lower_val:
            error -= sorted_in_d[len(X_train) - 1][2]
        else:
            error += sorted_in_d[len(X_train) - 1][2]
        if error < best_error:
            best_error = error
            best_threshold = sorted_in_d[len(X_train) - 1][0][d] + 1
            dim_for_h = d

    return best_threshold, best_error, dim_for_h


def updated_distrubtion(D, wt, ht, X_train, y_train):
    Zt = 0
    prod_vals = []
    h_pred, h_index, h_theta = ht
    for i in range(len(X_train)):
        h_val = h_pred if X_train[i][h_index] <= h_theta else -h_pred
        mult = D[i] * np.exp(-wt * y_train[i] * h_val)
        prod_vals.append(mult)
        Zt += mult
    new_D = np.array(prod_vals)
    new_D = (1 / Zt) * new_D
    return new_D


def get_error(weights, hypotheses, X, y):
    errors = 0
    for i in range(len(X)):
        if predict(X[i], hypotheses, weights) != y[i]:
            errors += 1
    return errors / len(X)


def get_exponential_loss(weights, hypotheses, X, y):
    e_vals = []
    for i in range(len(X)):
        weighted_sum = get_weighted_sum(X[i], hypotheses, weights)
        e_vals.append(np.exp(-y[i]*weighted_sum))
    return np.sum(e_vals)*(1/len(X))

def predict(x, hypotheses, weights):
    sum = 0
    for i in range(len(hypotheses)):
        h_pred, h_index, h_theta = hypotheses[i]
        weight = weights[i]
        h_val = h_pred if x[h_index] <= h_theta else -h_pred
        sum += weight * h_val
    return 1 if sum >= 0 else -1


def get_weighted_sum(x, hypotheses, weights):
    sum = 0
    for i in range(len(hypotheses)):
        h_pred, h_index, h_theta = hypotheses[i]
        h_val = h_pred if x[h_index] <= h_theta else -h_pred
        sum += weights[i] * h_val
    return sum


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    weights = []
    hypotheses = []

    distribution = np.array([1 / len(X_train) for i in range(len(X_train))])
    for i in range(1, T + 1):
        threshold_neg_1, error_neg_1, dim_neg_1 = get_weak_learner(distribution, X_train, y_train, -1)
        threshold_pos_1, error_pos_1, dim_pos_1 = get_weak_learner(distribution, X_train, y_train, 1)
        h_pred, h_index, h_theta = (-1, dim_neg_1, threshold_neg_1) if error_neg_1 < error_pos_1 else (
            1, dim_pos_1, threshold_pos_1)
        error = error_neg_1 if error_neg_1 < error_pos_1 else error_pos_1
        hypotheses.append((h_pred, h_index, h_theta))
        wt = 0.5 * np.log((1 - error) / error)
        weights.append(wt)

        distribution = updated_distrubtion(distribution, wt, hypotheses[-1], X_train, y_train)

    return hypotheses, weights


##############################################
# You can add more methods here, if needed.


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, 80)

    ##############################################
    # You can add more methods here, if needed.

    # section (a)
    training_errors = []
    test_errors = []
    # T = [i for i in range(1, 81)]

    for i in range(1, 81):
        training_errors.append(get_error(alpha_vals[:i], hypotheses[:i], X_train, y_train))
        test_errors.append(get_error(alpha_vals[:i], hypotheses[:i], X_test, y_test))

    # plt.plot(T, training_errors, color="blue")
    # plt.plot(T, test_errors, color="red")
    #
    # plt.xlabel("T (iterations)")
    # plt.ylabel("Error")
    #
    # plt.legend(["Training Error", "Test Error"])
    # plt.show()
    #

    # section (b)
    for i in range(10):
        h_pred, h_index, h_theta = hypotheses[i]
        print("AdaBoost iteration: " + str(i + 1) + " chooses word: " + vocab[
            h_index] + ", with h_pred: " + str(h_pred) + ", threshold value: " + str(h_theta) + ", weight:" + str(
            alpha_vals[i]))

    # section (c)
    training_loss = []
    test_loss = []
    #T = [i for i in range(1, 81)]


    for i in range(1, 81):
        training_loss.append(get_exponential_loss(alpha_vals[:i], hypotheses[:i], X_train, y_train))
        test_loss.append(get_exponential_loss(alpha_vals[:i], hypotheses[:i], X_test, y_test))

    # plt.plot(T, training_loss, color="blue")
    # plt.plot(T, test_loss, color="red")
    #
    # plt.xlabel("T (iterations)")
    # plt.ylabel("Loss")
    #
    # plt.legend(["Training Loss", "Test Loss"])
    # plt.show()


    ##############################################


if __name__ == '__main__':
    main()
