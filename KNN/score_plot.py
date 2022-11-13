import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from statistics import mode

import math 

import matplotlib.pyplot as plt
# Importing data to a pandas DataFrame


def score_plot(x_train, x_test, y_train, y_test):  # main function

    x_plot = np.arange(1, int(math.sqrt(len(y_test)))+1).tolist()
    y_plot_f1, y_plot_precision, y_plot_recall = [], [], []

    for i in range(int(math.sqrt(len(y_test)))):
        classifier = KNeighborsClassifier(n_neighbors=i+1, p=2, metric='euclidean')
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)

        y_plot_f1.append(f1_score(y_test, y_pred, average='micro'))
        y_plot_precision.append(precision_score(y_test, y_pred, average='micro'))
        y_plot_recall.append(recall_score(y_test, y_pred, average='micro'))



    plt.plot(x_plot, y_plot_f1, label='f1 score')
    plt.plot(x_plot, y_plot_precision, label='precision score')
    plt.plot(x_plot, y_plot_recall, label='recall score')
    plt.legend()
    plt.title('Accuracy rate of different K values')
    plt.grid()
    plt.show()

    a = [i for i, x in enumerate(y_plot_f1) if x == max(y_plot_f1)]
    b = [i for i, x in enumerate(y_plot_precision) if x == max(y_plot_precision)]
    c = [i for i, x in enumerate(y_plot_recall) if x == max(y_plot_recall)]

    mean = mode([a[-1], b[-1], c[-1]]) + 1
    if mean == 1:
        mean = 3
    if mean % 2 == 0:
        mean += 1
    optimal_k = mean
    print(f"K value used: {optimal_k}")
    
    return optimal_k
