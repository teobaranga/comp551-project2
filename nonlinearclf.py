import time
from operator import itemgetter

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split


def get_neighbours(train_set, test_set_element, k):
    """
    Get the K nearest neighbours of the test_set_element in the train_set
    """
    # calculate euclidean distance
    cp = (pairwise_distances(train_set, test_set_element)) ** 2
    # cp = train_set.copy()
    # cp.data -= np.take(np.tile(test_set.toarray()[0], np.diff(cp.indptr)[1]), cp.indices)
    # cp.data **= 2
    euc_distance = np.sqrt(np.sum(cp, axis=1))
    # return the index of nearest neighbour
    return np.argsort(euc_distance)[0:k]


def get_average_prediction(train_set_y, neighbour_indices):
    """
    Compute the kNN response by taking the most popular value from the neighbours
    """
    class_votes = {}
    for i in neighbour_indices:
        # Get the class of neighbour (y value)
        response = train_set_y[i]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def predict(train_x, train_y, test_x, k):
    # generate predictions
    test_y_pred = np.empty(test_x.shape[0])
    for i in range(test_x.shape[0]):
        neighbours = get_neighbours(train_x, test_x[i], k)
        test_y_pred[i] = get_average_prediction(train_y, neighbours)
    return test_y_pred


def predict_scikit(train_x, train_y, test_x, k):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_x, train_y)
    return neigh.predict(test_x)


def main():
    # Maximum number of rows to load from the dataset, or None to load all
    max_rows = 60000

    # Flag indicating whether to time the process or not
    running_time = True

    # Predict the Kaggle test set
    predict_test_set = False

    # Show statistics
    statistics = False

    # K neighbours
    k = 3

    # Load the training set
    # X: (Id, Text)
    train_set_x = np.genfromtxt('dat_train_x_no_eng.csv', delimiter=',', dtype=None, names=True, max_rows=max_rows,
                                usecols=[0, 1])
    # Y: (Id, Category)
    train_set_y = np.genfromtxt('dat_train_y_no_eng.csv', delimiter=',', dtype=None, names=True, max_rows=max_rows,
                                usecols=[0, 1])

    # Split the dataset into a training and test set:
    train_x, test_x, train_y, test_y = train_test_split(
        train_set_x["Text"], train_set_y["Category"],
        test_size=0.2, shuffle=True, random_state=551)

    # Use TF-IDF to model the data
    vec = TfidfVectorizer(analyzer='char')
    vec.fit(train_x)

    train_x = vec.transform(train_x)
    test_x = vec.transform(test_x)

    if running_time:
        start = time.time()

    # test_y_pred = predict_scikit(train_x, train_y, test_x, k)
    test_y_pred = predict(train_x, train_y, test_x, k)

    print("Training Set Accuracy:", (test_y == test_y_pred).mean())

    if predict_test_set:
        # Compute predictions for the real test set
        test_set_x = np.genfromtxt('test_set_x.csv', delimiter=',', dtype=None, names=True, usecols=[0, 1])
        test_x = vec.transform(test_set_x["Text"])
        test_y_pred = np.empty(test_x.shape[0])
        for i in range(test_x.shape[0]):
            neighbours = get_neighbours(train_x, test_x[i], k)
            test_y_pred[i] = get_average_prediction(train_y, neighbours)

        # Export CSV
        np.savetxt("test_set_y.csv", list(zip(test_set_x["Id"], test_y_pred)),
                   delimiter=",", fmt="%i", header="Id,Category", comments="")

    if running_time:
        print("Running time:", time.time() - start)

    if statistics:
        # Print the classification report
        print(metrics.classification_report(test_y, test_y_pred))

        # Plot the confusion matrix
        cm = metrics.confusion_matrix(test_y, test_y_pred)
        print(cm)

        import matplotlib.pyplot as plt

        plt.matshow(cm, cmap=plt.cm.jet)
        plt.show()


if __name__ == '__main__':
    main()
