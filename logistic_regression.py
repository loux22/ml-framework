import numpy as np 
from numpy import log,dot,exp,shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  
from ml.regression.logisitic import LogisticRegression

X,y = make_classification(n_features=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


regressor = LogisticRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

print(regressor.score(y_test,y_pred))
