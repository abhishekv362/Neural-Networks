import sklearn.datasets
breast_cancer = sklearn.datasets.load_breast_cancer()

# X and Y are numpy array
X = breast_cancer.data
Y = breast_cancer.target

# Converting Array to Dataframe
import pandas as pd
breast_cancer_df = pd.DataFrame( breast_cancer.data, columns = breast_cancer.feature_names)
breast_cancer_df['class'] = Y

from sklearn.model_selection import train_test_split

X = breast_cancer_df.drop('class', axis=1)
Y = breast_cancer_df['class']

# test_size specify % of total data we want as test case.
# stratify specify that the split in test and train should have equal distribution.
# random_state makes sure split made is same everytime we run the code, 1 is called seed value
# Seed must be between 0 and 2**32 - 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)

import matplotlib.pyplot as plt

# subplot(row,columns, fig No.)
plt.subplot(1,2,1)
plt.plot(X_train.T, '.')
plt.xticks(rotation = 'vertical')

# cut method of panadas convert the whole dataframe data into binary data in 1 go.
X_train_bin = X_train.apply(pd.cut, bins = 2, labels = [1, 0])
plt.subplot(1,2,2)
plt.plot(X_train_bin.T, '.')
plt.xticks(rotation = "vertical")
plt.ylim([-1,2])
plt.show()

# Choosing label as [1, 0] as from the graph shows the values in lower half of
# the plot contributes more towards malignant than above
X_test_bin = X_test.apply(pd.cut, bins = 2, labels = [1, 0])

# Coverting Data from DataFrame to Numpy Array
X_train_bin = X_train_bin.values
X_test_bin = X_test_bin.values

from sklearn.metrics import accuracy_score
import numpy as np

accuracy_list = {}
for b in range(X_train_bin.shape[1]+1):
	# Stores all the predictions
	prediction_list = []
	for x in X_train_bin:
		# Taking Sum of each row
		prediction = (np.sum(x) >= b)
		prediction_list.append(prediction)

	#Calculating accuracy for each value of 'b'
	accuracy = accuracy_score(prediction_list, Y_train)

	#Storing accuracy score for each value of 'b'
	accuracy_list[b] = accuracy
	print(b, accuracy)

best_b = max(accuracy_list, key = accuracy_list.get)
print(f"Best Value for Parameter is : {accuracy_list[best_b]} at {best_b}")
