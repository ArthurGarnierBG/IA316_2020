import numpy as np
class PureRandom:
    """ Random agent. """
    def __init__(self, nb_arms, seed=None):
        self._nb_arms = nb_arms
        self._rng = np.random.RandomState(seed)
        
    def act(self, context):
        minimum = np.argmin(np.array([len(context[0]),len(context[1])]))
        order = np.random.choice(len(context[minimum]),len(context[minimum]),replace = False)
        women=[]
        nope = []
        print(order)
        for i in order:
          b = context[2][i]
          c = np.random.choice(b,1)
          while c in nope:
            c =np.random.choice(b,1,replace=False)
            print(c)
          nope.append(c)
          women.append(c)
        action = [(order[i],women[i][0]) for i in range(len(context[minimum]))]
        return action
        
    def update(self, context, action, reward):
        pass