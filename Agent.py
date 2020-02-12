class Random:
    """ Random agent. """
    def __init__(self, nb_arms, seed=None):
        self._nb_arms = nb_arms
        self._rng = np.random.RandomState(seed)
        
    def act(self, context):
        action = self._rng.randint(len(context)) # note that action size is changing
        return action
        
    def update(self, context, action, reward):
        pass


