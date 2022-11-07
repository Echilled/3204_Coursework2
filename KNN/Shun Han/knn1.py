import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

import math
# Data processing
df = pd.read_csv("conbined_t1595_t1046.csv")

print(df.columns)

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])
    

x = df.iloc[:, 1:26]
y = df.iloc[:, 26]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=19, p=2, metric='euclidean')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(f"{classification_report(y_test, y_pred)}")