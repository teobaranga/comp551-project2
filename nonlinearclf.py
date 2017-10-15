from math import pow, sqrt
from operator import itemgetter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def euclidean_distance(a: csr_matrix, b: csr_matrix, length):
    """
    Compute the Euclidean distance between two points

    :param a: 1xN matrix (point)
    :param b: 1xN matrix (point)
    :param length: length/dimension of both a and b
    :return: the Euclidean distance
    """
    distance = 0
    for i in range(length):
        # Avoid computing the power for 0 elements
        if a[0, i] == 0 and b[0, i] == 0:
            continue
        distance += pow((a[0, i] - b[0, i]), 2)
    return sqrt(distance)


def get_neighbours(training_set_x: csr_matrix, training_set_y: csr_matrix, test_instance: csr_matrix, k: int) -> list:
    """
    Get the K nearest neighbours of the test_instance in the training_set_x

    :param training_set_x: MxN matrix
    :param training_set_y: Mx1 matrix
    :param test_instance: 1xN matrix
    :param k: number of nearest neighbours
    :return: a list containing the classes (y values) of the k nearest neighbours
    """
    distances = []
    length = test_instance.shape[1]
    for i in range(training_set_x.shape[0]):
        dist = euclidean_distance(training_set_x[i], test_instance, length)
        distances.append((training_set_y[i], dist))
    # sort by distance in increasing order
    distances.sort(key=itemgetter(1))
    neighbours = []
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours


def get_response(neighbours):
    """
    Compute the kNN response by taking the most popular value from the neighbours

    :param neighbours: list containing the classes for each neighbour
    :return: the most popular class
    """
    class_votes = {}
    for i in range(len(neighbours)):
        # Get the class of neighbour (y value)
        response = neighbours[i]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=itemgetter(1), reverse=True)
    return sorted_votes[0][0]


MAX_ROWS = 200

# Load the train set
# X: (Id, Text)
train_set_x = np.genfromtxt('dat_train_x_no_eng.csv', delimiter=',', dtype=None, names=True, max_rows=MAX_ROWS)
# Y: (Id, Category)
train_set_y = np.genfromtxt('dat_train_y_no_eng.csv', delimiter=',', dtype=None, names=True, max_rows=MAX_ROWS)

# Split the dataset in training and test set:
train_x, test_x, train_y, test_y = train_test_split(
    train_set_x["Text"], train_set_y["Category"],
    test_size=0.2, shuffle=True, random_state=551)

# Use TF-IDF to model the data
vec = TfidfVectorizer()
vec.fit(train_x)

train_x = vec.transform(train_x)
test_x = vec.transform(test_x)

# generate predictions
test_y_pred = np.empty(test_x.shape[0])
K = 3
for i in range(test_x.shape[0]):
    neighbours = get_neighbours(train_x, train_y, test_x[i], K)
    test_y_pred[i] = get_response(neighbours)

print("Training Set Accuracy:", (test_y == test_y_pred).mean())

# # Print the classification report
# print(metrics.classification_report(test_y, test_y_pred))
#
# # Plot the confusion matrix
# cm = metrics.confusion_matrix(test_y, test_y_pred)
# print(cm)
#
# import matplotlib.pyplot as plt
#
# plt.matshow(cm, cmap=plt.cm.jet)
# plt.show()
