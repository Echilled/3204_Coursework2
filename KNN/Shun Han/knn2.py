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
# Importing data to a pandas DataFrame
df = pd.read_csv("conbined_t1595_t1046.csv", thousands=',')


df['destination.port'] = df['destination.port'].astype('object')
df['source.port'] = df['source.port'].astype('object')
df['event.duration'] = df['event.duration'].astype(np.float64)

df = df.select_dtypes(include=np.number)

x = df.iloc[:, 1:8]
y = df.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

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


print(f"{classification_report(y_test, y_pred)}")