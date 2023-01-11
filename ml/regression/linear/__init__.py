from sklearn.datasets import make_regression
import numpy as np
from matplotlib import pyplot as plt

X, y = make_regression(n_samples=200, n_features=1, n_informative=1, noise=6, bias=30)


class LinearRegression():
    def __init__(self,X,m=200):
        ''' constructor '''
        self.X = X
        self.m = m
        
    def h(self,w):
        return (w[1]*np.array(self.X[:,0])+w[0])
  
    def cost(self, w, y):
        return (.5/self.m) * np.sum(np.square(self.h(w)-np.array(y)))

    def grad(self, w, y):
        g = [0]*2
        g[0] = (1/self.m) * np.sum(self.h(w)-np.array(y))
        g[1] = (1/self.m) * np.sum((self.h(w)-np.array(y))*np.array(self.X[:,0]))
        return g
  
    def descent(self, w_new, w_prev, lr):
        j=0
        while True:
            w_prev = w_new
            w0 = w_prev[0] - lr*self.grad(w_prev,y)[0]
            w1 = w_prev[1] - lr*self.grad(w_prev,y)[1]
            w_new = [w0, w1]
            if (w_new[0]-w_prev[0])**2 + (w_new[1]-w_prev[1])**2 <= pow(10,-6):
                return w_new
            if j>500: 
                return w_new
            j+=1 
          
    def graph(self, formula, x_range):  
        x = np.array(x_range)  
        y = formula(x)  
        plt.plot(x, y)
    
    def my_formula(self, x):
        w = [0,-1]
        w = self.descent(w,w,.1)
        return w[0]+w[1]*x
        
linear_reg = LinearRegression(X)

plt.scatter(X,y, c = "red",alpha=.5, marker = 'o')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.scatter(X,y, c = "red",alpha=.5, marker = 'o')
linear_reg.graph(linear_reg.my_formula, range(-5,5))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()