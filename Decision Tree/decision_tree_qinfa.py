import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def importdata():
    data = pd.read_csv("combined_qinfa.csv")

    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset obseravtions
    print("Dataset: ", data.head())

    data.replace({'-' : np.nan}, inplace=True)

    le = preprocessing.LabelEncoder()

    for col in data.columns:
        data[col] = le.fit_transform(data[col])
        data[col].replace({np.nan: data[col].median()}, inplace = True)

    return data


def train_using_gini(data):
    # Creating the classifier object
    # feature_cols = ['destination.port', 'source.port', 'event.duration']
    feature_cols = ['network.packets', 'source.port', 'source.bytes', 'destination.port']
    X = data[feature_cols]
    y = data.Technique

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    target_names = ['T1046', 'T1595', 'T1048']

    result = classification_report(y_test, y_predict, target_names=target_names, labels=[1, 2, 3])
    matrix = confusion_matrix(y_test, y_predict)
    report = [result, matrix]

    return report


# Function to perform training with entropy.
def train_using_entropy(data):
    # Decision tree with entropy
    # feature_cols = ['destination.port', 'source.port', 'event.duration']
    feature_cols = ['network.packets', 'source.port', 'source.bytes', 'destination.port']
    X = data[feature_cols]
    y = data.Technique

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    target_names = ['T1046', 'T1595', 'T1048']

    result = classification_report(y_test, y_predict, target_names = target_names, labels=[1, 2, 3])
    matrix = confusion_matrix(y_test, y_predict)
    report = [result, matrix]

    return report

# Driver code
def main():
    # Building Phase
    data = importdata()

    clf_gini = train_using_gini(data)

    clf_entropy = train_using_entropy(data)
    print("Results Using Entropy:")
    print(clf_entropy[0])
    print("Confusion matrix:")
    print(clf_entropy[1])
    print("\nResults Using Gini Index:")
    print(clf_gini[0])
    print("Confusion matrix:")
    print(clf_gini[1])


# Calling main function
if __name__ == "__main__":
    main()