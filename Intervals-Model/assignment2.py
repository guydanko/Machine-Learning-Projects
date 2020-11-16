#################################
# Your name: Guy Dankovich
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_values = np.random.uniform(0, 1, m)
        x_values = np.sort(x_values)
        y_values = []
        for x in x_values:
            y_values.append(self.get_y_distribution(x))
        y_values = np.array(y_values)
        return np.column_stack((x_values, y_values))

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        samples = self.sample_from_D(m)
        plt.scatter(samples[:, 0], samples[:, 1])
        plt.ylim(-0.1, 1.1)
        plt.axvline(0.2)
        plt.axvline(0.4)
        plt.axvline(0.6)
        plt.axvline(0.8)

        best_intervals = intervals.find_best_interval(samples[:, 0], samples[:, 1], k)[0]

        for interval in best_intervals:
            plt.hlines(y=-0.05, xmin=interval[0], xmax=interval[1], colors='r', linestyles='solid')
        plt.show()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        emp_error = []
        true_error = []
        for m in range(m_first, m_last + 1, step):
            emp_amount = 0
            true_amount = 0
            for i in range(T):
                sample = self.sample_from_D(m)
                best_intervals, error_amount = intervals.find_best_interval(sample[:, 0], sample[:, 1], 3)
                emp_amount += error_amount / m
                true_amount += self.caluclate_true_error(best_intervals)

            emp_error.append(emp_amount / T)
            true_error.append(true_amount / T)

        plt.plot([m for m in range(m_first, m_last + 1, step)], emp_error, color='red')
        plt.plot([m for m in range(m_first, m_last + 1, step)], true_error, color='blue')

        plt.legend(['Empirical Error', 'True Error'], loc='upper right')

        # plt.show()

        return np.column_stack((np.array(emp_error), np.array(true_error)))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        emp_errors = []
        true_errors = []
        min_erm_error = 1
        min_k = 0
        sample = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            best_intervals, error_amount = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            emp_error = error_amount / m
            emp_errors.append(emp_error)
            true_errors.append(self.caluclate_true_error(best_intervals))
            if emp_error < min_erm_error:
                min_erm_error = emp_error
                min_k = k

        plt.plot([k for k in range(k_first, k_last + 1, step)], emp_errors, color='red')
        plt.plot([k for k in range(k_first, k_last + 1, step)], true_errors, color='blue')

        plt.legend(['Empirical Error', 'True Error'], loc='upper right')

        # plt.show()

        return min_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        emp_errors = []
        true_errors = []
        penalties = []
        penalties_and_errors = []
        min_error_and_penalty = 1
        min_k = 0
        sample = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            best_intervals, error_amount = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            emp_error = error_amount / m
            true_error = self.caluclate_true_error(best_intervals)
            penalty = self.calc_penalty(0.1, m, k)
            emp_errors.append(emp_error)
            true_errors.append(true_error)
            penalties.append(penalty)
            error_sum = penalty + emp_error
            penalties_and_errors.append(error_sum)
            if error_sum < min_error_and_penalty:
                min_error_and_penalty = error_sum
                min_k = k

        plt.plot([k for k in range(k_first, k_last + 1, step)], emp_errors, color='red')
        plt.plot([k for k in range(k_first, k_last + 1, step)], true_errors, color='blue')
        plt.plot([k for k in range(k_first, k_last + 1, step)], penalties, color='green')
        plt.plot([k for k in range(k_first, k_last + 1, step)], penalties_and_errors, color='black')

        plt.legend(['Empirical Errors', 'True Errors', 'Penalties', 'Empirical Error + Penalties'], loc='upper left')

        # plt.show()

        return min_k

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        k_counts = [0 for k in range(11)]
        sample = self.sample_from_D(m)
        for i in range(T):
            np.random.shuffle(sample)
            training_set = [sample[i] for i in range(4 * m // 5)]
            training_set.sort(key=lambda x: x[0])
            holdout_set_x = [sample[i][0] for i in range(4 * m // 5, m)]
            holdout_set_y = [sample[i][1] for i in range(4 * m // 5, m)]
            min_k = 0
            min_error = 1
            for k in range(1, 11):
                best_intervals, error_amount = intervals.find_best_interval(
                    [training_set[i][0] for i in range(len(training_set))],
                    [training_set[i][1] for i in range(len(training_set))],
                    k)
                holdout_error = self.calc_holdout_error(best_intervals, holdout_set_x, holdout_set_y) / m
                if holdout_error < min_error:
                    min_error = holdout_error
                    min_k = k
            k_counts[min_k] += 1

        highest_freq = max(k_counts)
        return k_counts.index(highest_freq) + 1

    #################################
    # Place for additional methods
    def get_y_distribution(self, x):
        rand = np.random.uniform(0, 1, 1)
        if x <= 0.2 or 0.4 <= x <= 0.6 or x >= 0.8:
            if rand <= 0.8:
                return 1
            else:
                return 0
        else:
            if rand <= 0.1:
                return 1
            else:
                return 0

    def caluclate_true_error(self, intervals):
        inter1_1 = [0, 0.2]
        inter2_1 = [0.4, 0.6]
        inter3_1 = [0.8, 1]
        inter1_0 = [0.2, 0.4]
        inter2_0 = [0.6, 0.8]

        intersection_size_1 = 0

        for interval in intervals:
            intersection_size_1 += self.intersection_size(interval, inter1_1)
            intersection_size_1 += self.intersection_size(interval, inter2_1)
            intersection_size_1 += self.intersection_size(interval, inter3_1)

        intersection_size_0 = 0

        between_intervals = self.intervals_between(intervals)

        for interval in between_intervals:
            intersection_size_0 += self.intersection_size(inter1_0, interval)
            intersection_size_0 += self.intersection_size(inter2_0, interval)

        prob1but0error = (intersection_size_1 * 0.2) + (intersection_size_0 * 0.1)
        between_size = self.union_size(between_intervals)
        prob0but1error = (between_size - intersection_size_0) * 0.8 + (
                self.union_size(intervals) - intersection_size_1) * 0.9

        return prob0but1error + prob1but0error

    def intersection_size(self, interval1, interval2):
        beginning = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        if beginning > end:
            return 0
        return end - beginning

    def union_size(self, intervals):
        union = 0
        for interval in intervals:
            union += interval[1] - interval[0]
        return union

    def intervals_between(self, intervals):
        intervals.insert(0, (0, 0))
        intervals.append((1, 1))
        between = []
        for i in range(len(intervals) - 1):
            between.append((intervals[i][1], intervals[i + 1][0]))
        return between

    def calc_penalty(self, delta, m, k):
        return math.sqrt((8 / m) * (2 * k * math.log((2 * math.exp(1) * m) / (2 * k) + math.log(4 / delta))))

    def calc_holdout_error(self, intervals, holdoutX, holdoutY):
        errors = 0
        for i in range(len(holdoutX)):
            if self.is_in_interval(holdoutX[i], intervals) and holdoutY[i] == 0:
                errors += 1
            if (not self.is_in_interval(holdoutX[i], intervals)) and holdoutY[i] == 1:
                errors += 1
        return errors

    def is_in_interval(self, x, intervals):
        for interval in intervals:
            if interval[0] <= x <= interval[1]:
                return True
        return False
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    print(math.exp(1))
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
