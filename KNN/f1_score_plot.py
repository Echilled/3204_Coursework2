import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import math
import matplotlib.pyplot as plt


def f1_plot(x_train, x_test, y_train, y_test):
    x_plot = np.arange(1, int(math.sqrt(len(y_test)))+1).tolist()
    y_plot = []
    for i in range(int(math.sqrt(len(y_test)))):
        classifier = KNeighborsClassifier(n_neighbors=i+1, p=2, metric='euclidean')
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        y_plot.append(f1_score(y_test, y_pred))
    plt.plot(x_plot, y_plot)
    plt.grid()
    plt.show()


# print(f"{classification_report(y_test, y_pred)}")