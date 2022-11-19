import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import warnings

plt.rc("font", size=14)
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

# Reading of dataset file
data = pd.read_csv('Dataset 1.csv', header=0)
data = data.dropna()

# Data Preprocessing
data.replace({'-': np.nan}, inplace=True)  # converting '-' to NaN values

le = preprocessing.LabelEncoder()
for col in data.columns:
    if col != "Technique":
        # Encoding of values in columns
        data[col] = le.fit_transform(data[col])

        # replacing null values of each column with median value
        data[col].replace({np.nan: data[col].median()}, inplace=True)

# print(data.info())
# print(data.describe())
# print(data.head())


# Preparation of features set and target col set
features = data[list(data.columns)]
features.drop('Technique', inplace=True, axis=1)
features = features.to_numpy()  # converting feature set to numpy array
target = data['Technique'].to_numpy()  # converting target col to numpy array
# print(features.shape, len(target))  # shape of feature matrix and target vector


# Before Scaling Model Accuracy
initial_logreg = LogisticRegression(multi_class='multinomial', solver='sag')
initial_logreg.fit(features, target)
print('Initial Model Accuracy: ' + str(initial_logreg.score(features, target) * 100))


# Calculation on accuracy of data with standardization of data through scaling
m_pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='sag'))
m_pipe.fit(features, target)
print('Initial Model Accuracy after Scaling: ' + str(m_pipe.score(features, target) * 100))


# Calculation on accuracy of test dataset with standardization of data through scaling
# running train_test_split for our dataset
train_features, test_features, train_target, test_target = train_test_split(features, data['Technique'],
                                                                            test_size=0.3, random_state=42)
print("Size of Training Features:", train_features.shape)
print("Size of Training Targets:", train_target.shape)
print("Size of Testing Features:", test_features.shape)
print("Size of Testing Targets:", test_target.shape)

d_pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='sag'))
d_pipe.fit(test_features, test_target)
print('Model Accuracy on test dataset: ' + str(d_pipe.score(test_features, test_target) * 100))


print("\n===== Classification Report =====")
target_pred = d_pipe.predict(test_features)
print(classification_report(test_target, target_pred))


print("===== Confusion Matrix =====")
confusion_matrix = confusion_matrix(test_target, target_pred)
print(confusion_matrix)
