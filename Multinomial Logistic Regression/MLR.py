import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

plt.rc("font", size=14)
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

# Reading of dataset file
data = pd.read_csv('combined_t1595_t1046.csv', header=0)
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


# Preparation of features set and target col set
features = data[list(data.columns)]
features.drop('Technique', inplace=True, axis=1)
target = data['Technique']
# print(features.shape, len(target))  # shape of feature matrix and target vector


# Before Scaling Model Accuracy
initial_logreg = LogisticRegression(multi_class='multinomial', solver='sag')
initial_logreg.fit(features, target)
print('Initial Model Accuracy: \n\t> ' + str(initial_logreg.score(features, target) * 100))


# Performing Scaling on feature set
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

# Performing Feature Selection to remove redundant features
logreg = LogisticRegression(multi_class='multinomial', solver='sag')
logreg.fit(scaled_features, target)
coef = logreg.coef_[0]
redundant_features = list(pd.Series(scaled_features.columns)[list(coef == 0)])
n_scaled_features = scaled_features
for col_name in redundant_features:
    n_scaled_features.drop(col_name, inplace=True, axis=1)

# Calculation on accuracy of data with standard scaling and removal of redundant features
logreg = LogisticRegression(multi_class='multinomial', solver='sag')
logreg.fit(scaled_features, target)
print('Model Accuracy after standard scaling and removal of redundant features: \n\t> '
      + str(logreg.score(n_scaled_features, target) * 100))


# Calculation on accuracy of test dataset
# running train_test_split for our dataset
train_features, test_features, train_target, test_target = train_test_split(n_scaled_features, target,
                                                                            test_size=0.3, random_state=42)
print("Size of Training Features:", train_features.shape)
print("Size of Training Targets:", train_target.shape)
print("Size of Testing Features:", test_features.shape)
print("Size of Testing Targets:", test_target.shape)


t_logreg = LogisticRegression(multi_class='multinomial', solver='sag')
t_logreg.fit(test_features, test_target)
print('Model Accuracy on test dataset: \n\t> ' + str(t_logreg.score(test_features, test_target) * 100))


print("\n===== Classification Report on Test Dataset =====")
target_pred = t_logreg.predict(test_features)
print(classification_report(test_target, target_pred))

print("===== Confusion Matrix on Test Dataset =====")
confusion_matrix = confusion_matrix(test_target, target_pred)
print(confusion_matrix)

# plotting the Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix)
ax.grid(False)
ax.set_xlabel('Predicted outputs', fontsize=14, color='black')
ax.set_ylabel('Actual outputs', fontsize=14, color='black')
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='white')
plt.savefig('Confusion Matrix Diagram.png')
