from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import optimal_k
import matplotlib.pyplot as plt


def readlogFile(file):
    df = pd.read_csv(file)
    return df


def check_int(value):
    try:
        int(value)
        return value
    except ValueError:
        return np.NaN


def check_valid_string(value):
    try:
        int(value)
        return np.NaN
    except ValueError:
        return value

def format_columns_preprocessing(dataframe):
    le = LabelEncoder()
    machine_properties = ['host.ip', 'host.mac',
                          'destination.ip', 'destination.mac']
    trunc_columns = ['source.packets', 'source.bytes', 'destination.packets', 'destination.bytes',
                     'network.packets', 'network.bytes', 'event.duration', 'destination.port']
    dataframe = dataframe.replace(to_replace='[(K?B),]', value='', regex=True)
    dataframe = dataframe.replace('-', 0)
    for column in trunc_columns:
        # Add column and fill it with zeroes if it does not exist
        dataframe[column] = dataframe.get(column, 0)
        dataframe[column] = dataframe[column].apply(check_int)
        dataframe[column] = dataframe[column].astype(np.number)
    for column in machine_properties:
        dataframe[column] = dataframe.get(column, '')
        dataframe[column] = dataframe[column].apply(check_valid_string)
        dataframe[column] = le.fit_transform(dataframe[column])


    dataframe = dataframe.fillna(0)
    result = dataframe['Technique']
    x_columns = pd.concat(
        [dataframe[trunc_columns], dataframe[machine_properties]], axis=1)
    dataframe = pd.concat([x_columns, result], axis=1)

    return dataframe


def knn_train(knn, X_train, y_train):
    knn.fit(X_train, y_train)


def knn_predict(knn, X_test):
    pred = knn.predict(X_test)
    return pred


def real_time_processing(csv, scaling=True):
    df = readlogFile(csv)
    df = format_columns_preprocessing(df)
    if scaling is True:
        scaler = StandardScaler()
        scaler.fit(df.drop('Technique', axis=1))
        scaled_features = scaler.transform(df.drop('Technique', axis=1))
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, df['Technique'], test_size=0.3, random_state=42)
        realtimeX_test = X_test
        return realtimeX_test
    elif scaling is False:
        non_scaled = df.iloc[:, 1:-2]
        X_train, X_test, y_train, y_test = train_test_split(
            non_scaled, df['Technique'], test_size=0.30, random_state=42)
        realtimeX_test = X_test
        return realtimeX_test


def main(k=None, realtime=None):
    global logs_source
    dataframe = readlogFile("..\Consistent_logs\combined_t1595_t1570_t1020.csv")
    dataframe = shuffle(dataframe)
    dataframe = format_columns_preprocessing(dataframe)
    # Standardize variables using scaling
    scaler = StandardScaler()
    scaler.fit(dataframe.drop('Technique', axis=1))
    scaled_features = scaler.transform(dataframe.drop('Technique', axis=1))

    non_scaled = dataframe.iloc[:, 1:-2]
    # Training of test split data, testing size is 30 percent
    # Evaluate model scaled version
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, dataframe['Technique'], test_size=0.30, random_state=42)
    samples = dataframe['Technique']
    opt_k = optimal_k.optimal_k_plot(
        X_train, X_test, y_train, y_test, samples) if k is None else k
    knn = KNeighborsClassifier(n_neighbors=opt_k)
    knn_train(knn, X_train, y_train)
    if realtime == 'Y':
        logs_source = input("Enter log file for real time testing:")
        realtime_test = real_time_processing(logs_source)
        X_test = realtime_test
        prediction = knn_predict(knn, X_test)
        print(f"K value used: {opt_k}")
        pred_results = Counter(list(prediction.flatten()))
        att_type = list(pred_results.keys())
        att_count = list(pred_results.values())
        plt.bar(att_type, att_count)
        plt.xlabel("Attack Techniques")
        plt.ylabel("Count")
        plt.title("Attack Techniques detected: Scaled")
        plt.show()
    else:
        prediction = knn_predict(knn, X_test)
        print(f"K value used: {opt_k}")
        print(
            f"Classification report for scaled input:\n{classification_report(y_test, prediction)}")
        print(
            f"Confusion matrix:\n{multilabel_confusion_matrix(y_test, prediction)}")
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            display_labels=dataframe['Technique'].unique(),
            cmap=plt.cm.Blues
        )
        disp.ax_.set_title("Confusion Matrix")
        plt.savefig('Confusion Matrix.png')
        plt.show()

    # Evaluate model non scaled version
    X_train, X_test, y_train, y_test = train_test_split(
        non_scaled, dataframe['Technique'], test_size=0.30, random_state=42)
    opt_k = optimal_k.optimal_k_plot(
        X_train, X_test, y_train, y_test, samples) if k is None else k
    knn = KNeighborsClassifier(n_neighbors=opt_k)
    knn_train(knn, X_train, y_train)
    if realtime == 'Y':
        realtime_test = real_time_processing(logs_source, scaling=False)
        X_test = realtime_test
        prediction = knn_predict(knn, X_test)
        print(f"K value used: {opt_k}")
        pred_results = Counter(list(prediction.flatten()))
        att_type = list(pred_results.keys())
        att_count = list(pred_results.values())
        plt.bar(att_type, att_count)
        plt.xlabel("Attack Techniques")
        plt.ylabel("Count")
        plt.title("Attack Techniques detected: Non-scaled")
        plt.show()
    else:
        prediction = knn_predict(knn, X_test)
        print(f"K value used: {opt_k}")
        print(
            f"Classicaition report for non-scaled input:\n{classification_report(y_test, prediction)}")
        print(
            f"Confusion matrix:\n{multilabel_confusion_matrix(y_test, prediction)}")
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            display_labels=dataframe['Technique'].unique(),
            cmap=plt.cm.Blues
        )
        disp.ax_.set_title("Confusion Matrix")
        plt.savefig('Confusion Matrix.png')
        plt.show()


if __name__ == "__main__":
    realtime = input("Running Real time? Default No (Y/N):")
    k = input("Enter K value (return key to calculate optimal k value): ")
    if realtime.upper() == 'Y' and k.isnumeric():
        main(int(k), 'Y')
    elif k.isnumeric() and realtime.upper() != 'Y':
        main(int(k))
    elif realtime == 'Y':
        main(None, 'Y')
    else:
        print("Running program with optimal k value")
        main()
