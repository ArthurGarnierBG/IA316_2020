import numpy as np
class Pure_Random_Agent:
    """ Random agent. """
    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def act(self, men_class, women_class, possible_recommendation, user_match_history):
        context = [men_class, women_class, possible_recommendation]
        minimum = np.argmin(np.array([len(context[0]),len(context[1])]))
        order = np.random.choice(len(context[minimum]),len(context[minimum]),replace = False)
        women=[]
        nope = []
        for i in order:
          b = context[2][i]
          for el in nope:
            if el in b:
              b.remove(el)
          if len(b)>=1:
            c = np.random.choice(b,1)
            nope.append(c)
            women.append(c)

          nope.append(c)
          women.append(c)
        action = [(order[i],women[i][0]) for i in range(len(context[minimum]))]
        return action

    def update(self, rewards, recommended_pairs, men_class, women_class):
        pass
