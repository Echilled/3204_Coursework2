
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
import optimal_k


def readlogFile(file):
    df = pd.read_csv(file)
    return df


def format_columns_preprocessing(dataframe):
    le = preprocessing.LabelEncoder()
    for column in dataframe.columns:
        if column != 'Technique':
            dataframe[column] = le.fit_transform(dataframe[column])


def knn_algo(X_train, X_test, y_train, k_value):
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return pred


def knn_train(knn, X_train, y_train):
    knn.fit(X_train, y_train)


def knn_predict(knn, X_test):
    pred = knn.predict(X_test)
    return pred


def real_time_processing(csv):
    df = readlogFile(csv)
    format_columns_preprocessing(df)
    scaler = StandardScaler()
    scaler.fit(df.drop('Technique', axis=1))
    scaled_features = scaler.transform(df.drop('Technique', axis=1))
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Technique'], test_size=0.3)
    realtimeX_test = X_test
    return realtimeX_test


def main():
    dataframe = readlogFile("..\Consistent_logs\conbined_t1595_t1046.csv")
    dataframe = shuffle(dataframe)
    format_columns_preprocessing(dataframe)
    # Standardize variables using scaling
    scaler = StandardScaler()
    scaler.fit(dataframe.drop('Technique', axis=1))
    scaled_features = scaler.transform(dataframe.drop('Technique', axis=1))

    non_scaled = dataframe.iloc[:, 1:-2]
    # Training of test split data, testing size is 30 percent
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, dataframe['Technique'], test_size=0.30)
    samples = dataframe['Technique']
    opt_k = optimal_k.optimal_k_plot(X_train, X_test, y_train, y_test, samples)
    knn = KNeighborsClassifier(n_neighbors=opt_k)
    knn_train(knn, X_train, y_train)
    # realtime_test = real_time_processing('..\Consistent_logs\conbined_t1595_t1046.csv')
    prediction = knn_predict(knn, X_test)

    # prediction = knn_algo(X_train, X_test, y_train, opt_k)

    # Evaluate model
    # print(confusion_matrix(y_test, prediction))
    print(f"Classification report for scaled input:\n{classification_report(y_test, prediction)}")

    # X_train, X_test, y_train, y_test = train_test_split(non_scaled, dataframe['Technique'], test_size=0.30)
    # opt_k = optimal_k.optimal_k_plot(X_train, X_test, y_train, y_test, samples)
    # knn = KNeighborsClassifier(n_neighbors=opt_k)
    # knn_train(knn, X_train, y_train)
    # prediction = knn_predict(knn, X_test)

    # print(confusion_matrix(y_test, prediction))
    # print(f"Classicaition report for non-scaled input:\n{classification_report(y_test, prediction)}")


if __name__ == "__main__":
    main()
