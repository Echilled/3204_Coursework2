import math

import pandas as pd
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from KNN import optimal_k
from KNN import score_plot
import numpy as np
import matplotlib.pyplot as plt


def readlogFile(file):
    df = pd.read_csv(file)
    return df


def format_columns_preprocessing(dataframe):
    le = preprocessing.LabelEncoder()
    for column in dataframe.columns:
        if column != 'Technique':
            dataframe[column] = le.fit_transform(dataframe[column])


def knn_algo(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return pred


def main():
    dataframe = readlogFile("tailored_logs/T1595-T1570-T1020_Packetbeat_raw_Gp16_SimYewSiangMerrill-SimKaiChing-RachelWongSiHui-YeoHanJordan.csv")
    format_columns_preprocessing(dataframe)

    # Standardize variables using scaling
    scaler = StandardScaler()
    scaler.fit(dataframe.drop('Technique', axis=1))
    scaled_features = scaler.transform(dataframe.drop('Technique', axis=1))

    # Training of test split data, testing size is 30 percent
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, dataframe['Technique'], test_size=0.30)
    prediction = knn_algo(X_train, X_test, y_train, y_test)
    optimal_k.optimal_k_plot(X_train, X_test, y_train, y_test)
    score_plot.score_plot(X_train, X_test, y_train, y_test)
    # Evaluate model
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))


if __name__ == "__main__":
    main()