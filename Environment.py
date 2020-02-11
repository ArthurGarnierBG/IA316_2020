import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import random
from scipy.stats import norm
import networkx as nx
from collections import defaultdict



def random_argmax(rng, list_):
    """ similar to np.argmax but return a random element among max
        when multiple max exists."""
    return rng.choice(np.argwhere(list_ == list_.max()).flatten())

class ExplicitFeedback:
    """ A rating environment with explicit feedback.
        User and items are represented by points in R^k
        User interest for a given item is modeled by a parametric function
        R_{u,i} = f(u,i) = f(W_u, W_i)
        Example of function include dot product (cosine similarity)
        R_{u,i} = \sum_k w_{u,k} . w_{i,k}
        action: Recommend one item for a given user among those he has never bought before
    """

    def __init__(self, nb_users_men=30, nb_users_women=30, 
                 internal_embedding_size=10,
                 seed=None):
        
        self.nb_users_men = nb_users_men
        self.nb_users_women = nb_users_women
        self.internal_embedding_size = internal_embedding_size
        self._rng = np.random.RandomState(seed)
        #What about the left users?
        self.action_size = min(nb_users_men, nb_users_women)
        self.sampling_limit = nb_users_men * nb_users_women
        self.men_mean = np.ones(self.internal_embedding_size)
        self.men_var = np.ones(self.internal_embedding_size)
        self.women_mean = np.ones(self.internal_embedding_size)
        self.women_var = np.ones(self.internal)
        self.men_embedding = None
        self.women_embedding = None
        self.users_history = None
        self.z_cut_points = None
        self.done = False

    def step(self, action):
        # check if behind done
        if self.done: #self.user_item_history.sum() >= self.sampling_limit:
            print("You are calling step after it return done=True.\n"
                  "You should reset the environment.")

        assert action < self.action_size
        self.action = action
        
        # compute potential rewards
        #potential_rewards = [self._get_user_item_rating(self.current_user, i) 
        #                     for i in np.argwhere(self.user_item_history[self.current_user, :] == 0).flatten()]

        potential_rewards = [[self._get_user_match(j,i) 
                             for i in np.argwhere(self.user_match_history[j, :] == 0).flatten()] for j in self.nb_users_men]

    	graph = nx.from_numpy_matrix(potential_rewards)
    	pairs = graph.edges()


    	#matching_pairs = [pairs[0]]
    	#for p in pairs[1:]:
    #		if p not in matching_pairs and len(matching_pairs)<nb_users_men: ## a voir si cardinaux egaux
    		#	matching_pairs.append(p)

    	#potential_max_match = [pairs[0]]
    	#for i,j in pairs:
    #		if ((i and j) not in potential_max_match[:,0]) and ((i and j) not in potential_max_match[:,1]):
    #			potential_max_match.append((i,j))


    	a = defaultdict(int)
    	s=0
   
    	for i,j in pairs:
    		a[i]+=1
    	for key,d in a.items():
    			s+=d-1

        optimal_return = len(pairs)-s #self.action_size

        # map action to item
        for p in action:
        	self.user_match_history[p[0],p[1]] =1 
        	self.user_match_history[p[1],p[0]] =1 
        #self.recommended_item = np.argwhere(self.user_match_history[self.current_user, :] == 0)[action][0]

        # mark item as rated
        #self.user_match_history[self.current_user, self.recommended_item] = 1

        # compute reward R_t
        self.current_match = [self._get_user_item_rating(p[0],p[1]) for p in action]
        self.reward = np.sum(self.current_match)

             # check if done
        if self.user_match_history.sum() == self.sampling_limit:
            self.done = True

        # compute next state S_{t+1}
        self._next_state()

        # update action space t+1
        self.action_size = len(self.available_items)

        return self.reward, self.state, self.done, optimal_return

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self.action_size = min(self.nb_users_men, self.nb_users_women)
        
        # create users and items embedding matrix
        self.men_embedding = self._rng.normal(loc=self.men_mean,
                                                scale=self.men_var,
                                                size=(self.nb_users_men, self.internal_embedding_size))
        self.women_embedding = self._rng.normal(loc=self.women_mean,
                                                scale=self.women_var,
                                                size=(self.nb_users_women, self.internal_embedding_size))

        # Let X = users_embedding and Y = items_embedding
        # In order to properly map float into integers, we need to know the distribution of
        # Z = \sum_k X_k.Y_k
        # E[Z] = \sum_k E[X_k.Y_k] = \sum_k E[X_k]E[Y_k]
        # Var[Z] = \sum_k Var[X_k.Y_k] = \sum_k Var[X_k]Var[Y_k] + Var[X_k]E[Y_k]^2 + Var[Y_k]E[X_k]^2
        z_mean = self.men_mean.dot(self.women_mean)
        z_var = self.men_var.dot(self.women_var) + self.men_var.dot(np.square(self.women_mean)) + \
                self.women_var.dot(np.square(self.men_mean))
        z = norm(z_mean, np.sqrt(z_var))
        # to get 5 values, we need 4 cut points
               self.z_cut_points = z.ppf([0.8]) # you can control the distribution of ratings here.
        self.user_match_history = np.zeros((self.nb_users_men, self.nb_users_women))
        self.done = False

        self._next_state()
        return self.state

    def _get_user_match(self, user1, user2):
        real_score = self.men_embedding[user1].dot(self.women_embedding[user2])
        match_score = np.searchsorted(self.z_cut_points, real_score)
        return match_score

    def _get_variables(self, user, item):
        user_embedding = self.users_embedding[user]
        item_embedding = self.items_embedding[item]
        if self.displayed_users_embedding_size + self.displayed_items_embedding_size > 0:
            variables = np.array([user_embedding[:self.displayed_users_embedding_size],
                                  item_embedding[:self.displayed_items_embedding_size]])

            if self.noise_size > 0:
                noise = self._rng.normal(loc=np.ones(self.noise_size),
                                         scale=np.ones(self.noise_size),
                                         size=self.noise_size)
                variables = np.append(variables, noise)

            return variables

    def _get_new_user(self):
        for i in range(10):
            user = self._rng.randint(0, self.nb_users)
            # check it remain at least one item
            if np.sum(self.user_match_history[user, :]) < self.nb_items:
                return user

        return self._rng.choice(np.argwhere(self.user_match_history == 0))[0]

    def _next_state(self):
        # Pick a user
        if self.user_match_history.sum() < self.sampling_limit:
            self.current_user = self._get_new_user()
        else:
            self.current_user = None

        # List available items
        self.available_items = np.argwhere(self.user_match_history[self.current_user, :] == 0)

        self.state = list()
        for i in self.available_items:
            item = i[0]
            # Compute variables
            variables = self._get_variables(self.current_user, item)
            self.state.append([self.current_user, item, variables])
