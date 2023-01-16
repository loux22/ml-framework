import numpy as np 
from numpy import log,dot,exp,shape

class LogisticRegression:
    def __init__(self, lr=0.001,epochs=400):
        self.lr = lr
        self.epochs = epochs

    def fit(self,X,y):
        weights,X = self.initialize(X)
        cost_list = np.zeros(self.epochs,)
        for i in range(self.epochs):
            weights = weights - self.lr*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = self.cost(weights, X, y)
        self.weights = weights
        return cost_list
    
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis

    def score(self,y,y_hat):
        tp,tn,fp,fn = 0,0,0,0
        for i in range(len(y)):
            if y[i] == 1 and y_hat[i] == 1:
                tp += 1
            elif y[i] == 1 and y_hat[i] == 0:
                fn += 1
            elif y[i] == 0 and y_hat[i] == 1:
                fp += 1
            elif y[i] == 0 and y_hat[i] == 0:
                tn += 1
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*precision*recall/(precision+recall)
        return f1_score
    
    # fonction sigmoid qui va determiner le seuil de classification
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig

    # initialisation du poid et de X
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    
    # algo d'optimisation -> descent gradiant
    def cost(self, theta, X, y):
        z = dot(X,theta)
        cost0 = y.T.dot(log(self.sigmoid(z)))
        cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
        cost = -((cost1 + cost0))/len(y)
        return cost


