import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

breast_cancer = sklearn.datasets.load_breast_cancer()
breast_cancer_DF = pd.DataFrame( breast_cancer.data, columns = breast_cancer.feature_names)
breast_cancer_DF['class'] = breast_cancer.target
X = breast_cancer_DF.drop('class', axis =1)
Y = breast_cancer_DF['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, stratify = Y, random_state = 1)
X_train_binarized = X_train.apply( pd.cut, bins = 2, labels= [1,0])
X_test_binarized = X_test.apply( pd.cut, bins = 2, labels= [1,0])
X_train_binarized = X_train_binarized.values
X_test_binarized = X_test_binarized.values

accuracy_list = {}
for b in range(X_train_binarized.shape[1]+1):
  prediction_list = []
  for x in X_train_binarized:
    prediction = (np.sum(x) >= b)
    prediction_list.append(prediction)
  accuracy = accuracy_score(prediction_list, Y_train)
  accuracy_list[b] = accuracy
  print(b, accuracy)

key_found = max(accuracy_list, key = accuracy_list.get)
print(f"Best Accuracy at {key_found} : {accuracy_list[key_found]}" )
