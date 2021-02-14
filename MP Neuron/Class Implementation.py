import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

# Loading Data
breast_cancer = sklearn.datasets.load_breast_cancer()

#Converting Data from Numpy Array to pandas DataFrame
breast_cancer_DF = pd.DataFrame( breast_cancer.data, columns = breast_cancer.feature_names)

#Adding a 'class' column
breast_cancer_DF['class'] = breast_cancer.target

#Dropping 'class' column
X = breast_cancer_DF.drop('class', axis =1)

#Storing class column in variable 'Y'
Y = breast_cancer_DF['class']

# Creating Training and Test Cases from available Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, stratify = Y, random_state = 1)

# Converting Trained and Test data into Binary data
X_train_binarized = X_train.apply( pd.cut, bins = 2, labels= [1,0])
X_test_binarized = X_test.apply( pd.cut, bins = 2, labels= [1,0])

# Coverting Data from DataFrame to Numpy Array
X_train_binarized = X_train_binarized.values
X_test_binarized = X_test_binarized.values

# Defining a MP Nueron Model Class
class MPNeuron:
  def __init__(self):
    self.b = None

  def model(self, x):
    return (sum(x) >= self.b)

  def predict(self, X):
    prediction_list = []
    for x in X:
      prediction = self.model(x)
      prediction_list.append(prediction)
    return (np.array(prediction_list))

  def fit(self, X, Y):
    accuracy = {}

    for b in range(X.shape[1]+1):
      self.b = b;
      Y_predicted = self.predict(X)
      accuracy[b] = accuracy_score(Y_predicted, Y)

    best_b = max(accuracy, key = accuracy.get)
    self.b = best_b

    print("Optimal value of b is :", best_b)
    print("Heighest accuracy is :", accuracy[best_b])


# Creating an instance of MPNeuron Class
FirstInstance = MPNeuron()

# Calling the Model For Training
FirstInstance.fit(X_train_binarized, Y_train)

# Calling the Model For Testing
FirstInstance.fit(X_test_binarized, Y_test)

# Note ::
# It might be possible that accuracy is greater for other value of ′b′ in test−data
# than what we got from training−data but we only do testing on the value that yield maximum in training
# here b=28