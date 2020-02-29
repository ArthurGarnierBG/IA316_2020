from scipy.optimize import linear_sum_assignment
import numpy as np

class Smart_Random:
    """ Smart random agent. Each possible recommendation has a random weight, then Hungarian algorithm to determine the optimized assignment"""
    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def act(self, men_embedding, women_embedding, possible_recommendation):
        weight = np.zeros((len(men_embedding), len(women_embedding)))
        

        return action

    def update(self, context, action, reward):
        pass
