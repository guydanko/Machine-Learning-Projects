from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy.random

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def predict_image(images, labels, query, k):
    distances = []
    for i in range(len(images)):
        distances.append((numpy.linalg.norm(images[i] - query), labels[i]))
    distances.sort()
    return get_predicted_digit(distances, k)


def get_predicted_digit(distances, k):
    digit_count = [0 for i in range(10)]
    for i in range(k):
        digit_count[int(distances[i][1])] += 1
    max_count = 0
    for i in range(10):
        if digit_count[i] > max_count:
            max_count = digit_count[i]
            max_digit = i
    return max_digit


# answer for (b)
testCorrect = 0
for i in range(len(test)):
    if int(test_labels[i]) == predict_image(train[:1000], train_labels[:1000], test[i], 10):
        testCorrect += 1

print(testCorrect / len(test))

# answer for (c)
correct_percent = [0 for i in range(100)]
for k in range(1, 101):
    for i in range(len(test)):
        sum_correct = 0
        if int(test_labels[i]) == predict_image(train[:1000], train_labels[:1000], test[i], k):
            sum_correct += 1
    correct_percent[k - 1] = sum_correct / len(test)

plt.plot([i for i in range(1, 101)], correct_percent)
plt.show()

# answer for (d)
correct_percent = [0 for i in range(50)]
k = 0
for n in range(100, 5001, 100):
    sum_correct = 0
    for i in range(len(test)):
        if int(test_labels[i]) == predict_image(train[:n], train_labels[:n], test[i], 1):
            sum_correct += 1
    correct_percent[k] = sum_correct / len(test)
    k += 1

plt.plot([i for i in range(100, 5001, 100)], correct_percent)
plt.show()
