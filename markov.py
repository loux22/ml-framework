from ml.markov import Markov
import numpy as np

M = np.array([[0.5,0.25,0,0.25],[0.25,0.5,0,0],[0,0.25,1,0.25],[0.25,0,0,0.5]])
P = np.array([[1],[0],[0],[0]])

markov = Markov(M, P)
markov.proba(0.99,2)
print(markov.NIteration(1))