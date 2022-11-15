from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math

from statistics import median


k_counter = 1


def optimal_k_plot(X_train, X_test, y_train, y_test, samples):
    error_rate = []
    for i in range(1, int(math.sqrt(len(samples))) + 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, int(math.sqrt(len(samples))) + 1), error_rate, color='blue',
             linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

    return compare_k(X_train, X_test, y_train, y_test, error_rate)


def compare_k(X_train, X_test, y_train, y_test, error_rate):
    global k_counter
    _f1, _precision, _recall, _accuracy = [], [], [], []

    low_error = [i for i, x in enumerate(error_rate) if x == min(error_rate)]  # populate list with index of k value where error rate is the lowest
    for i in low_error:
        classifier = KNeighborsClassifier(
            n_neighbors=i+1, p=2, metric='euclidean')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        _f1.append(f1_score(y_test, y_pred, average='micro'))
        _precision.append(precision_score(y_test, y_pred, average='micro'))
        _recall.append(recall_score(y_test, y_pred, average='micro'))
        _accuracy.append(accuracy_score(y_test, y_pred))

    # populate lists with index of k value where scores are the highest
    a = [i+k_counter for i, x in enumerate(_f1) if x == max(_f1)]
    b = [i+k_counter for i, x in enumerate(_precision) if x == max(_precision)]
    c = [i+k_counter for i, x in enumerate(_recall) if x == max(_recall)]
    d = [i+k_counter for i, x in enumerate(_accuracy) if x == max(_accuracy)]

    res = a + b + c + d
    res.sort()
    k_counter += len(low_error)
    _median = int(median(res))  # get the median k value out of 4 comparison results
    if _median == 1:  # if k value is too low e.g., 1 (too much noise), recalculate next best k value
        _median = [x for i, x in enumerate(
            error_rate) if i not in low_error if i >= len(low_error)]
        _median = len(low_error) + compare_k(X_train,
                                             X_test, y_train, y_test, _median)
    if _median % 2 == 0:
        _median += 1
    optimal_k = _median
    return optimal_k