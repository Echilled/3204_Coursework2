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


def linearPredict(featureMat, weights, biases, k):
    """This is the linear predictor function for out MLR model. It calculates the logit scores for each possible outcome.

    Args-
        featureMat- A numpy array of features
        weights- A numpy array of weights for our model
        biases- A numpy array of biases for our model

    Returns-
        logitScores- Logit scores for each possible outcome of the target variable for each feature set in the feature matrix
    """
    logitScores = np.array(
        [np.empty([k]) for i in range(featureMat.shape[0])])  # creating empty(garbage value) array for each feature set

    for i in range(featureMat.shape[0]):  # iterating through each feature set
        # calculates logit score for each feature set then flattens the logit vector
        logitScores[i] = (weights.dot(featureMat[i].reshape(-1, 1)) + biases).reshape(-1)

    return logitScores


def softmaxNormalizer(logitMatrix, k):
    """Converts logit scores for each possible outcome to probability values.

    Args-
        logitMatrix - This is the output of our logitPredict function; consists  logit scores for each feature set

    Returns-
        probabilities - Probability value of each outcome for each feature set
    """

    probabilities = np.array([np.empty([k]) for i in
                              range(logitMatrix.shape[0])])  # creating empty(garbage value) array for each feature set

    for i in range(logitMatrix.shape[0]):
        exp = np.exp(logitMatrix[i])  # exponentiates each element of the logit array
        sumOfArr = np.sum(exp)  # adds up all the values in the exponentiated array
        probabilities[i] = exp / sumOfArr  # logit scores to probability values
    return probabilities


def multinomialLogReg(features, weights, biases, k):
    """Performs logistic regression on a given feature set.

    Args-
        features- Numpy array of features(standardized)
        weights- A numpy array of weights for our model
        biases- A numpy array of biases for our model

    Returns-
        probabilities, predictions
        Here,
            probabilities: Probability values for each possible outcome for each feature set in the feature matrix
            predictions: Outcome with max probability for each feature set
    """
    logitScores = linearPredict(features, weights, biases, k)
    probabilities = softmaxNormalizer(logitScores, k)
    predictions = np.array([np.argmax(i) for i in probabilities])  # returns the outcome with max probability
    return probabilities, predictions


def crossEntropyLoss(probabilities, target):
    """Calculates cross entropy loss for a set of predictions and actual targets.

    Args-
        predictions- Probability predictions, as returned by multinomialLogReg function
        target- Actual target values
    Returns-
        CELoss- Average cross entropy loss
    """
    n_samples = probabilities.shape[0]
    CELoss = 0
    for sample, i in zip(probabilities, target):
        CELoss += -np.log(sample[i])
    CELoss /= n_samples
    return CELoss


def stochGradDes(learning_rate, epochs, target, features, weights, biases, k):
    """Performs stochastic gradient descent optimization on the model.

    Args-
        learning_rate- Size of the step the function will take during optimization
        epochs- No. of iterations the function will run for on the model
        target- Numpy array containing actual target values
        features- Numpy array of independent variables
        weights- Numpy array containing weights associated with each feature
        biases- Array containinig model biases

    Returns-
        weights, biases, loss_list
        where,
            weights- Latest weight calculated (Numpy array)
            bias- Latest bias calculated (Numpy array)
            loss_list- Array containing list of losses observed after each epoch
    """
    target = target.astype(int)
    loss_list = np.array([])  # initiating an empty array

    for i in range(epochs):
        probabilities, _ = multinomialLogReg(features, weights,
                                             biases, k)  # Calculates probabilities for each possible outcome
        CELoss = crossEntropyLoss(probabilities,
                                  target)  # Calculates cross entropy loss for actual target and predictions
        loss_list = np.append(loss_list, CELoss)  # Adds the CELoss value for the epoch to loss_list

        probabilities[np.arange(features.shape[0]), target] -= 1  # Substract 1 from the scores of the correct outcome

        grad_weight = probabilities.T.dot(features)  # gradient of loss w.r.t. weights
        grad_biases = np.sum(probabilities, axis=0).reshape(-1, 1)  # gradient of loss w.r.t. biases

        # updating weights and biases
        weights -= (learning_rate * grad_weight)
        biases -= (learning_rate * grad_biases)

    return weights, biases, loss_list


# Reading of dataset file
data = pd.read_csv('Dataset 1.csv', header=0)
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

logitTest = linearPredict(features, weights, biases, target_k)
probabilities, predictions = multinomialLogReg(features, weights, biases, target_k)


initial_logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
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

logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
logreg.fit(train_features, train_target)

target_pred = logreg.predict(test_features)
print('Model Accuracy on test dataset after standard scaling: ' + str(logreg.score(test_features, test_target) * 100))


updatedWeights, updatedBiases, loss_list = stochGradDes(0.1, 2000, train_target, train_features,
                                                        weights, biases, target_k)
testProbabilities, testPredictions = multinomialLogReg(test_features, updatedWeights, updatedBiases, target_k)

print("\n===== Classification Report =====")
print(classification_report(test_target, target_pred))

confusion_matrix = confusion_matrix(test_target, target_pred)
print("===== Confusion Matrix =====")
print(confusion_matrix)
