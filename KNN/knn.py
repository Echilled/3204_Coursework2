import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as p


def readlogFile(file):
    df = pd.read_csv(file)
    return df


def format_columns_preprocessing(dataframe):
    le = preprocessing.LabelEncoder()
    for column in dataframe.columns:
        dataframe[column] = le.fit_transform(dataframe[column])


def main():
    dataframe = readlogFile('tailored_logs/T1595_PacketbeatTraffic_clean_Gp14_EugeneChew.csv')
    format_columns_preprocessing(dataframe)

    # Standardize variables using scaling #
    scaler = StandardScaler()
    scaler.fit(dataframe.drop('Binary', axis=1))
    # scaled_features = scaler.transform(dataframe.drop('Binary', axis=1))


if __name__ == "__main__":
    main()