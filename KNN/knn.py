import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
import numpy as p


def readlogFile(file):
    df = pd.read_csv(file)
    return df


def main():
    dataframe = readlogFile('../T1595ActiveScanning_logs/T1595_PacketbeatTraffic_clean_Gp14_EugeneChew.csv')
    print(dataframe)


if __name__ == "__main__":
    main()