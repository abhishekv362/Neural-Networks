import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Loading Data
breast_cancer = sklearn.datasets.load_breast_cancer()

# Converting Data from Numpy Array to pandas DataFrame
breast_cancer_DF = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Adding a 'class' column
breast_cancer_DF['class'] = breast_cancer.target

# Dropping 'class' column
X = breast_cancer_DF.drop('class', axis=1)

# Storing class column in variable 'Y'
Y = breast_cancer_DF['class']

# Creating Training and Test Cases from available Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Coverting Data from DataFrame to Numpy Array
X_train = X_train.values
X_test = X_test.values


# Defining Perceptron Model Class
class Perceptron:
	def __init__(self):
		self.b = None
		self.w = None

	def model(self, x):
		return (1 if np.dot(x, self.w) >= self.b else 0)

	def predict(self, X):
		prediction_list = []
		for x in X:
			prediction = self.model(x)
			prediction_list.append(prediction)
		return np.array(prediction_list)

	# 2 Introduction of 1st hyperparameter "Epoch", whereas w and x are parameters
	# 2 Epoch : A single pass over all the data points + labels while training the model.
	def fit(self, X, Y, epochs = 1):
		accuracy = {}
		self.w = np.ones(X.shape[1])
		self.b = 0

		accuracy = {}
		max_accuracy = 0


		for iter in range(epochs):
			for x, y in zip(X, Y):
				Y_predicted = self.model(x)
				if Y_predicted == 0 and y == 1:
					# 4 Introduction of 2nd Hyper-parameter "Learning Rate"
					self.w = self.w + x
					self.b = self.b + 1
				elif Y_predicted == 1 and y == 0:
					self.w = self.w - x
					self.b = self.b - 1

			accuracy[iter] = accuracy_score(self.predict(X), Y)

			if accuracy[iter] > max_accuracy:
				max_accuracy = accuracy[iter]

				# 3 Introduction Of Checkpoints
				# Stores the best prediction , to test upon the model
				chkpt_w = self.w
				chkpt_b = self.b

		self.w = chkpt_w
		self.b = chkpt_b

		plt.plot(list(accuracy.values()))
		plt.ylim([0,1])
		plt.show()
		max_accuracy_epoch = (max(accuracy, key=accuracy.get))
		print(f"Max at {max_accuracy_epoch} value is {accuracy[max_accuracy_epoch]}")


"""
Note ::
What is the downside of using too large of a learning rate?
The loss tends to oscillate around a local minima, without converging at the lowest point.
"""

# Creating an instance of Perceptron Class
perceptron = Perceptron()
perceptron.fit(X_train, Y_train, 100)
