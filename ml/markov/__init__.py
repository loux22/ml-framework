import numpy as np
from random import *
import matplotlib.pyplot as plt

class Markov:

    def __init__(self, matrice_transition, position) :
        self.matrice_transition = matrice_transition
        self.position = position
    
    # calcul proba de la position pour n fois l'evenement
    def NIteration(self, n) :
        for k in range(n) :
            self.position = self.matrice_transition.dot(self.position)
        return self.position
    
    # calcul en combien de temps le pourcentage de chance choisis(percent) que la position soit à l'arrivé
    def proba(self, percent, expected_position):
        time = 0
        while self.position[expected_position,0] < percent :
            self.position  = self.matrice_transition.dot(self.position)
            time += 1
        print(time)

        
