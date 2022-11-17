import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

plt.rc("font", size=14)
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")


# function for standardizing data
def standardScaler(feature_array):
    """Takes the numpy.ndarray object containing the features and performs standardization on the matrix.
    The function iterates through each column and performs scaling on them individually.

    Args:
        feature_array- Numpy array containing training features

    Returns:
        None
    """

    total_cols = feature_array.shape[1]  # total number of columns
    for i in range(total_cols):  # iterating through each column
        feature_col = feature_array[:, i]
        mean = feature_col.mean()  # mean stores mean value for the column
        std = feature_col.std()  # std stores standard deviation value for the column
        feature_array[:, i] = (feature_array[:, i] - mean) / std  # standard scaling of each element of the column


# Reading of dataset file
data = pd.read_csv('Main_Dataset.csv', header=0)
data = data.dropna()

# Data Preprocessing
data.replace({'-': np.nan}, inplace=True)  # converting '-' to NaN values

le = preprocessing.LabelEncoder()
for col in data.columns:
    # Encoding of values in columns
    data[col] = le.fit_transform(data[col])

    # replacing null values of each column with median value
    data[col].replace({np.nan: data[col].median()}, inplace=True)

# print(data.info())
# print(data.describe())
# print(data.head())
# for i in data.columns:
#     print(data[i].value_counts())

# Standardization of data
features = data[list(data.columns)]
features = features.to_numpy()  # converting feature set to numpy array
target = data['Technique'].to_numpy()  # converting target col to numpy array
# print(features.shape, len(target))  # shape of feature matrix and target vector

standardScaler(features)  # performing standardization on feature set

# # checking if standarization worked
# total_cols = features.shape[1]  # total number of columns
# for i in range(total_cols):
#     print(features[:, i].std())

target_k = len(data['Technique'].unique())
feature_m = len(list(data.columns))

# creating randomized weights for our linear predictor func
weights = np.random.rand(target_k, feature_m)
# creating randomized biases for our linear predictor func
biases = np.random.rand(target_k, 1)

initial_logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
initial_logreg.fit(features, target)
print('Initial Model Accuracy: ' + str(initial_logreg.score(features, target) * 100))

# running train_test_split for our dataset
train_features, test_features, train_target, test_target = train_test_split(features, data['Technique'],
                                                                            test_size=0.3, random_state=42)
standardScaler(train_features)  # standard scaling training set
standardScaler(test_features)  # standard scaling testing set
print("Size of Training Features:", train_features.shape)
print("Size of Training Targets:", train_target.shape)
print("Size of Testing Features:", test_features.shape)
print("Size of Testing Targets:", test_target.shape)

logreg = LogisticRegression(multi_class='multinomial', solver='sag')
logreg.fit(train_features, train_target)

target_pred = logreg.predict(test_features)
print('Model Accuracy on test dataset after standard scaling: ' + str(logreg.score(test_features, test_target) * 100))


print("\n===== Classification Report =====")
print(classification_report(test_target, target_pred))

confusion_matrix = confusion_matrix(test_target, target_pred)
print("===== Confusion Matrix =====")
print(confusion_matrix)
