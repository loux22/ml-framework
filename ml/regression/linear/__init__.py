# y = w * x + b 
# y = valeur dépendante estimée.
# b = constante ou biais (ordonnée à l'origine).
# w = coefficient de régression ou pente (poids).
# x = valeur de la variable indépendante. 

# OBJECTIF : trouver l'ensemble optimal de paramètres qui minimise la fonction de perte -> la valeur optimale de la pente (m) et de la constante (b). 
# gradient descent -> algorithme d'optimisation

import numpy as np


class LinearRegression() :
    def __init__(self, epochs = 1000, learning_rate = 0.01) :
        self.learning_rate = learning_rate
        self.epochs = epochs

    # fontion d'entrainement
    def fit(self, X, Y) :
        self.m, self.n = X.shape
        # initialisation du poids
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = Y
        # apprentissage gradient descent 
        for i in range(self.epochs) :
            self.update_weights()
        return self

    def predict(self, X) :
        # y = w * x + b
        return X.dot(self.w) + self.b
    
    # calcul du score
    def score(self, X_test, y_test):
        sst = np.sum((X_test - X_test.mean())**2)
        ssr = np.sum((y_test - X_test)**2)
        r2 = 1-(sst/ssr)
        return r2
    

    # fonction gradient descent -> En utilisant la descente de gradient, nous calculons de manière itérative les gradients de la fonction de perte par rapport aux paramètres 
    # et continuons de mettre à jour les paramètres jusqu'à ce que nous atteignions le minimum. 
    def update_weights(self) :
        Y_pred = self.predict(self.X)
        # calcul gradient
        dW = - (2 * (self.X.T).dot(self.y - Y_pred) ) / self.m
        db = - 2 * np.sum(self.y - Y_pred) / self.m
        # mise à jours poids et biais
        self.w = self.w - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self