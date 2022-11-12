import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

from statistics import mode

import matplotlib.pyplot as plt
# Importing data to a pandas DataFrame

def main():
    df = pd.read_csv("../tailored_logs/conbined_t1595_t1046.csv", thousands=',')

    result = df['Technique']

    df['destination.port'] = df['destination.port'].astype('object')
    df['source.port'] = df['source.port'].astype('object')
    df['event.duration'] = df['event.duration'].astype(np.float64)

    df = df.select_dtypes(include=np.number)
    df = pd.concat([df, result], axis=1)


    x = df.iloc[:, 1:8]
    y = df.iloc[:, 8]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    x_plot = np.arange(1, int((len(y_test)))+1).tolist()
    y_plot_f1, y_plot_precision, y_plot_recall = [], [], []

    for i in range(int((len(y_test)))):
        classifier = KNeighborsClassifier(n_neighbors=i+1, p=2, metric='euclidean')
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)

        y_plot_f1.append(f1_score(y_test, y_pred, pos_label='T1595'))
        y_plot_precision.append(precision_score(y_test, y_pred, pos_label='T1595'))
        y_plot_recall.append(recall_score(y_test, y_pred, pos_label='T1595'))


    plt.plot(x_plot, y_plot_f1, label='f1 score')
    plt.plot(x_plot, y_plot_precision, label='precision score')
    plt.plot(x_plot, y_plot_recall, label='recall score')
    plt.legend()
    plt.grid()
    plt.show()

    a = y_plot_f1.index(max(y_plot_f1))
    b = y_plot_precision.index(max(y_plot_precision))
    c = y_plot_recall.index(max(y_plot_recall))

    return mode([a, b, c]) + 1

