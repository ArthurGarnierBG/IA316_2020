from scipy.optimize import linear_sum_assignment
import numpy as np

class Smart_Random_Agent:
    """ Smart random agent. Each possible recommendation has a random weight, then Hungarian algorithm to determine the optimized assignment"""
    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def act(self, men_embedding, women_embedding, possible_recommendation):
        weight = np.zeros((len(men_embedding), len(women_embedding)))
        man = 0
        for man_rec in possible_recommendation:
            for woman in man_rec:
                weight[man][woman] = -np.random.uniform(0,1)
            man += 1

        col, row = linear_sum_assignment(weight)
        recommendation = [(col[i], row[i]) for i in range(len(col))]
        #What if rectangular matrix of weights ?
        print("Agent recommendation pair: "+str(recommendation))
        return(recommendation)

    def update(self, reward):
        pass
