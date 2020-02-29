import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import random
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment


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
        self.user_match_history = None
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
        print("Potential rewards :"+str(potential_rewards))


        #Let's compute the optimal number of good recommendation
        cost = -potential_rewards
        row, col = linear_sum_assignment(cost)
        optimal_reward = -cost[row, col].sum()
        print("optimal reward :"+str(optimal_reward))

        # map couple as already recommended
        left_app = 0
        index_left_app = []
        for p in action:
            self.user_match_history[p[0],p[1]] = 1
            #We won't recommend couples that left the app
            if(self._get_user_match(p[0],p[1])) == 4:
                left_app += 1
                self.user_match_history[p[0],:] = 1
                self.user_match_history[:,p[1]] = 1
                index_left_app.append((p[0], p[1]))

        # compute reward R_t
        self.current_match = [self._get_user_match(p[0],p[1]) for p in action]
        self.reward = np.sum(self.current_match)

        #Compute the number of men and women starting using the left_app
        new_user_man = np.random.randint(left_app+2)
        new_user_woman = np.random.randint(left_app+2)

        self.update_new_users(new_user_man, new_user_woman, index_left_app)

        print("User match history : "+str(self.user_match_history))

        # check if done
        if self.user_match_history.sum() == self.sampling_limit:
            self.done = True

        #Compute the possible recommendations based on the match history
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :] ==0).flatten()] for j in range(self.nb_users_men)]

        return self.reward, self.men_embedding, self.women_embedding, self.possible_recommendation, self.done, optimal_reward

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self.action_size = min(self.nb_users_men, self.nb_users_women)

        # create users and items embedding matrix
        self.men_embedding = self.get_new_user_men(self.nb_users_men)
        self.women_embedding = self.get_new_user_women(self.nb_users_women)

        z_mean = self.men_mean.dot(self.women_mean)
        z_var = self.men_var.dot(self.women_var) + self.men_var.dot(np.square(self.women_mean)) + \
                self.women_var.dot(np.square(self.men_mean))
        z = norm(z_mean, np.sqrt(z_var))
        self.z_cut_points = z.ppf([0.5, 0.9]) # you can control the distribution of matches here.
        self.user_match_history = np.zeros((self.nb_users_men, self.nb_users_women))
        self.done = False

        #Compute the possible recommendation based on the match history
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :] ==
            0).flatten()] for j in range(self.nb_users_men)]

        return self.men_embedding, self.women_embedding, self.possible_recommendation

    def _get_user_match(self, user1, user2):
        real_score = self.men_embedding[user1].dot(self.women_embedding[user2])
        #print("real score: "+str(real_score))
        match_score = np.searchsorted(self.z_cut_points, real_score)
        #print("Match score: "+str(match_score))
        match_score = match_score**2
        return match_score

    #Return features for new users
    def get_new_user_men(self, nb_users_men):
        return self._rng.normal(loc=self.men_mean, scale=self.men_var, size=(nb_users_men, self.internal_embedding_size))

    def get_new_user_women(self, nb_users_women):
        return self._rng.normal(loc=self.men_mean, scale=self.men_var, size=(nb_users_women, self.internal_embedding_size))

    #Update embeddings and user_match_history
    def update_new_users(self, new_user_man, new_user_woman, index_left_couple):
        #Compute indices of men and women leaving the app
        left_man_index = [index_left_couple[i][0] for i in range(len(index_left_couple))]
        left_woman_index = [index_left_couple[i][1] for i in range(len(index_left_couple))]
        #Delete left users from embeddings and history
        self.user_match_history = np.delete(self.user_match_history, left_man_index, 0)
        self.user_match_history = np.delete(self.user_match_history, left_woman_index, 1)
        self.men_embedding = np.delete(self.men_embedding, left_man_index, 0)
        self.women_embedding = np.delete(self.women_embedding, left_woman_index, 0)
        #Update nb of men and women
        self.nb_users_men -= len(index_left_couple)
        self.nb_users_women -= len(index_left_couple)

        if(new_user_man > 0):
            man_embedding = self.get_new_user_men(new_user_man)
            self.men_embedding = np.append(self.men_embedding, man_embedding, axis=0)
            self.user_match_history = np.append(self.user_match_history, [np.zeros(self.nb_users_women)]*new_user_man, axis=0)
            self.nb_users_men += new_user_man

        if(new_user_woman > 0):
            woman_embedding = self.get_new_user_men(new_user_woman)
            self.women_embedding = np.append(self.women_embedding, woman_embedding, axis=0)
            self.user_match_history = np.append(self.user_match_history, [np.zeros(new_user_woman)]*self.nb_users_men, axis=1)
            self.nb_users_women += new_user_woman



if __name__ == "__main__":
    env = TinderEnv()
    men_embedding, women_embedding, possible_recommendation = env.reset(seed=2020)
    print(np.array(men_embedding).shape)
    print(np.array(women_embedding).shape)
    print(np.array(env.user_match_history).shape)
    print(np.array(possible_recommendation).shape)
    print("Men embedding:"+str(men_embedding))
    print("Woman embedding:"+str(women_embedding))
    print(np.array(env.user_match_history))
    print(possible_recommendation)


    recommendation = np.array([(0,0), (1,3), (2,2), (3,1)])

    reward, men_embedding, women_embedding, possible_recommendation, done, optimal_reward = env.step(recommendation)

    print("possible recommendation : "+str(possible_recommendation))

    print('reward: ', reward)
