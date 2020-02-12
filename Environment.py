import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import random
from scipy.stats import norm
import networkx as nx
from collections import defaultdict


class TinderEnv: 

    def __init__(self, nb_users_men=4, nb_users_women=4, 
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
        self.women_var = np.ones(self.internal_embedding_size)
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

        assert len(action) <= self.action_size
        self.action = action
        
        # compute potential rewards
        potential_rewards = np.array([[self._get_user_match(j,i) for i in
            np.argwhere(self.user_match_history[j, :] == 0).flatten()] for j in
            range(self.nb_users_men)])
        print(potential_rewards)
        #Let's compute the optimal number of good recommendation
        graph = nx.from_numpy_matrix(potential_rewards)
        pairs = graph.edges()
        a = defaultdict(int)
        s=0
        for i,j in pairs:
            a[i]+=1
        for key,d in a.items():
            s+=d-1

        optimal_return = len(pairs)-s
        print(optimal_return)
        # map couple as already recommended
        for p in action:
            self.user_match_history[p[0],p[1]] = 1 

        # compute reward R_t
        self.current_match = [self._get_user_match(p[0],p[1]) for p in action]
        self.reward = np.sum(self.current_match)

        # check if done
        if self.user_match_history.sum() == self.sampling_limit:
            self.done = True
        
        #Compute the possible recommendations based on the match history
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :]
            ==0).flatten()] for j in range(self.nb_users_men)]

        return self.reward, self.men_embedding, self.women_embedding, self.possible_recommendation, self.done, optimal_return

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

        z_mean = self.men_mean.dot(self.women_mean)
        z_var = self.men_var.dot(self.women_var) + self.men_var.dot(np.square(self.women_mean)) + \
                self.women_var.dot(np.square(self.men_mean))
        z = norm(z_mean, np.sqrt(z_var))
        self.z_cut_points = z.ppf([0.5]) # you can control the distribution of matches here.
        self.user_match_history = np.zeros((self.nb_users_men, self.nb_users_women))
        self.done = False
        
        #Compute the possible recommendation based on the match history 
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :] ==
            0).flatten()] for j in range(self.nb_users_men)]

        return self.men_embedding, self.women_embedding, self.possible_recommendation

    def _get_user_match(self, user1, user2):
        real_score = self.men_embedding[user1].dot(self.women_embedding[user2])
        match_score = np.searchsorted(self.z_cut_points, real_score)
        return match_score


if __name__ == "__main__":
    env = TinderEnv()
    men_embedding, women_embedding, possible_recommendation = env.reset(seed=2020)
    print(np.array(men_embedding).shape)
    print(np.array(women_embedding).shape)
    print(np.array(possible_recommendation).shape)
    print(men_embedding)
    print(women_embedding)
    print(possible_recommendation)

    recommendation = np.array([(0,0), (1,2), (2,1), (3,3)])

    reward, men_embedding, women_embedding, possible_recommendation, done, optimal_return = env.step(recommendation)
    
    print('reward: ', reward)


