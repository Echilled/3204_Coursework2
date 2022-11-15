from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import math

from statistics import median

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


def compare_k(X_train, X_test, y_train, y_test, error_rate):  # main function
    y_plot_f1, y_plot_precision, y_plot_recall, y_plot_accuracy = [], [], [], []

    # for i in range(int(math.sqrt(len(y_test)))):
    low_error = [i for i, x in enumerate(error_rate) if x == min(error_rate)]
    for i in range(len(low_error)):
        classifier = KNeighborsClassifier(n_neighbors=i+1, p=2, metric='euclidean')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        y_plot_f1.append(f1_score(y_test, y_pred, average='micro'))
        y_plot_precision.append(precision_score(y_test, y_pred, average='micro'))
        y_plot_recall.append(recall_score(y_test, y_pred, average='micro'))
        y_plot_accuracy.append(accuracy_score(y_test, y_pred))

    a = [i+1 for i, x in enumerate(y_plot_f1) if x == max(y_plot_f1)]
    b = [i+1 for i, x in enumerate(y_plot_precision) if x == max(y_plot_precision)]
    c = [i+1 for i, x in enumerate(y_plot_recall) if x == max(y_plot_recall)]
    d = [i+1 for i, x in enumerate(y_plot_accuracy) if x == max(y_plot_accuracy)]

    res = a + b + c + d
    res.sort()

    med = int(median(res))
    if med == 1:
        med = error_rate.index(median(error_rate))
    if med % 2 == 0:
        med += 1
    optimal_k = med
    print(f"K value used: {optimal_k}")
    return optimal_k

def main():
    print('hi')


if __name__ == "__main__":
    main()