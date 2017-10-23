import random
import time
from operator import itemgetter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, pairwise_distances, f1_score
from sklearn.model_selection import KFold


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
        response = train_set_y.iloc[i]
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
    max_rows = 100000

    # Flag indicating whether to time the process or not
    running_time = False

    # Predict the Kaggle test set
    predict_test_set = False

    # Show statistics
    statistics = False

    # Use scikit
    scikit_knn = False

    # K neighbours
    k = 9

    random_state = 551

    # Load the training set, loading a random portion of a specific size
    n = 276517  # number of records in file (excludes header)
    random.seed(12345)
    skip = sorted(random.sample(range(1, n + 1), n - max_rows))  # exclude the 0-indexed header
    # X: (Id, Text)
    train_set_x = pd.read_csv('train_set_x.csv', encoding='utf-8', keep_default_na=False, skiprows=skip)
    # Y: (Id, Category)
    train_set_y = pd.read_csv('train_set_y.csv', encoding='utf-8', keep_default_na=False, skiprows=skip)

    # Use TF-IDF to model the data
    vec = TfidfVectorizer(analyzer='char')

    if running_time:
        start = time.time()

    min_k = k
    max_k = k

    # creating odd list of K for KNN
    neighbors = list(filter(lambda x: x % 2 != 0, list(range(min_k, max_k + 1))))

    # empty list that will hold f1 scores
    f1_scores = []

    n_fold = 10

    for k in neighbors:
        # Split the dataset into a training and test set (K-fold):
        kf = KFold(n_splits=n_fold, random_state=random_state, shuffle=True)

        score = 0

        for train_index, test_index in kf.split(train_set_x):
            train_x, test_x = train_set_x["Text"][train_index], train_set_x["Text"][test_index]
            train_y, test_y = train_set_y["Category"][train_index], train_set_y["Category"][test_index]

            vec.fit(train_x)

            train_x = vec.transform(train_x)
            test_x = vec.transform(test_x)

            if not predict_test_set:
                if scikit_knn:
                    test_y_pred = predict_scikit(train_x, train_y, test_x, k)
                else:
                    test_y_pred = predict(train_x, train_y, test_x, k)

                print("Training Set Accuracy:", (test_y == test_y_pred).mean())

                score = score + f1_score(test_y, test_y_pred, average='weighted')
                print(score)

                if statistics:
                    # Plot the confusion matrix
                    cm = confusion_matrix(test_y, test_y_pred)
                    sns.heatmap(cm)
                    plt.show()

            break

            # f1_scores.append(score / n_fold)

    # # determining best k
    # optimal_k = neighbors[f1_scores.index(max(f1_scores))]
    # print("The optimal number of neighbors is %d" % optimal_k)
    #
    # # plot F1 scores error vs k
    # plt.plot(neighbors, f1_scores)
    # plt.xlabel('Number of Neighbors K')
    # plt.ylabel('F1 scores')
    # plt.show()

    if predict_test_set:
        # Compute predictions for the real test set
        test_set_x = pd.read_csv('test_set_x.csv', encoding='utf-8', keep_default_na=False)
        test_x = vec.transform(test_set_x["Text"])
        if scikit_knn:
            test_y_pred = predict_scikit(train_x, train_y, test_x, k)
        else:
            test_y_pred = predict(train_x, train_y, test_x, k)

        # Export CSV
        np.savetxt("test_set_y.csv", list(zip(test_set_x["Id"], test_y_pred)),
                   delimiter=",", fmt="%i", header="Id,Category", comments="")

    if running_time:
        print("Running time:", time.time() - start)


if __name__ == '__main__':
    main()
