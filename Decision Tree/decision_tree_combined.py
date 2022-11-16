import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def importdata():
    data = pd.read_csv("combined_t1595_t1046.csv")

    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    # Printing the dataset obseravtions
    print("Dataset: ", data.head())
    return data


def train_using_gini(data):
    # Creating the classifier object
    feature_cols = ['destination.port', 'source.port', 'event.duration']
    X = data[feature_cols]
    y = data.Technique

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    target_names = ['T1046', 'T1595']
    result = classification_report(y_test, y_predict, target_names=target_names)

    return result


# Function to perform training with entropy.
def train_using_entropy(data):
    # Decision tree with entropy
    feature_cols = ['destination.port', 'source.port', 'event.duration']
    X = data[feature_cols]
    y = data.Technique

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    target_names = ['T1046','T1595']
    result = classification_report(y_test, y_predict, target_names = target_names)

    return result

# Driver code
def main():
    # Building Phase
    data = importdata()

    clf_gini = train_using_gini(data)

    clf_entropy = train_using_entropy(data)
    print("Results Using Entropy:")
    print (clf_entropy)
    print("Results Using Gini Index:")
    print (clf_gini)


# Calling main function
if __name__ == "__main__":
    main()