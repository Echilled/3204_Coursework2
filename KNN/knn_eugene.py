import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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

    # Standardize variables using scaling
    scaler = StandardScaler()
    scaler.fit(dataframe.drop('Binary', axis=1))
    scaled_features = scaler.transform(dataframe.drop('Binary', axis=1))
    dataframe_feat = pd.DataFrame(scaled_features, columns=dataframe.columns[:-1])
    print(dataframe_feat.head())


if __name__ == "__main__":
    main()