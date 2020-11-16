import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import fetch_lfw_people
import numpy as np
from sklearn.model_selection import train_test_split


def plot_vector_as_image(image, h, w):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title("Ariel Sharon", size=12)
    plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if (target == target_label):
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
              of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """

    covMatrix = np.dot(np.transpose(X), X)
    u, s, ut = np.linalg.svd(covMatrix, full_matrices=True)
    U = np.array([ut[i] for i in range(k)])
    S = np.array([s[i] for i in range(k)])
    return U, S


def sectionB():
    selected_images, h, w = get_pictures_by_name()
    X = np.array(selected_images)[:, :, 0]
    U, S = PCA(X, 10)
    for i in range(len(U)):
        plot_vector_as_image(U[i], h, w)


def sectionC():
    selected_images, h, w = get_pictures_by_name()
    X = np.array(selected_images)[:, :, 0]
    mean = X.mean(axis=0)
    X = X - mean
    kVals = [1, 5, 10, 30, 50, 100]
    L2norms = []
    for k in kVals:
        U, S = PCA(X, k)
        reducePic = np.dot(U, np.transpose(X))
        recoverPic = np.dot(np.transpose(U), reducePic)
        recoverPic = np.transpose(recoverPic)
        sumDistance = 0
        randPic = np.random.randint(0, len(X) - 1, size=5)
        for i in randPic:
            plot_vector_as_image(X[i], h, w)
            plot_vector_as_image(recoverPic[i], h, w)
            sumDistance += np.linalg.norm(X[i] - recoverPic[i])
        L2norms.append(sumDistance)

    plt.plot([k for k in kVals], L2norms, color='blue')
    plt.xlabel('K')
    plt.ylabel('L2 distances')
    plt.title('L2 distances for K values')
    plt.show()


def sectionD():
    data = load_data()
    dataNames = data.target_names
    allPictures = []
    yVals = []
    hVals = []
    wVals = []
    for name in dataNames:
        pics_by_name, h, w = get_pictures_by_name(name)
        allPictures += pics_by_name
        yVals += [name] * len(pics_by_name)
        hVals += [h] * len(pics_by_name)
        wVals += [w] * len(pics_by_name)

    X = np.array(allPictures)[:, :, 0]
    mean = X.mean(axis=0)
    X = X - mean

    X_train, X_test, y_train, y_test = train_test_split(X, yVals, test_size=0.25, random_state=0)

    kVals = [1, 5, 10, 30, 50, 100, 150, 300]
    predictionScores = []
    for k in kVals:
        U, S = PCA(X, k)
        reducedTrainingPics = np.transpose(np.dot(U, np.transpose(X_train)))
        reducedTestPics = np.transpose(np.dot(U, np.transpose(X_test)))
        svmClassifier = svm.SVC(kernel='rbf', C=1000, gamma=10 ** -7).fit(reducedTrainingPics, y_train)
        predictionScores.append(svmClassifier.score(reducedTestPics, y_test))

    plt.plot([k for k in kVals], predictionScores, color='blue')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for K values')
    plt.show()


if __name__ == '__main__':
    sectionD()
