import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs

class TinderEnv:

    def __init__(self,
                nb_users_men=150,
                nb_users_women=150,
                nb_classes = 15,
                internal_embedding_size=10,
                mega_score = 5,
                matching_score= 1,
                std = 0.01,
                seed=None):

        self.std = std
        self.nb_classes = nb_classes
        self.nb_users_men = nb_users_men
        self.nb_users_women = nb_users_women
        self.internal_embedding_size = internal_embedding_size
        self._rng = np.random.RandomState(seed)
        self.action_size = min(self.nb_users_men, self.nb_users_women)
        self.sampling_limit = self.nb_users_men * self.nb_users_women
        self.men_mean = np.ones(self.internal_embedding_size)
        self.men_var = np.ones(self.internal_embedding_size)
        self.women_mean = np.ones(self.internal_embedding_size)
        self.women_var = np.ones(self.internal_embedding_size)
        self.mega_score = mega_score
        self.matching_score = matching_score
        self.men_embedding = None
        self.men_class = None
        self.women_embedding = None
        self.women_class = None
        self.X = None
        self.y = None
        self.user_match_history = None
        self.user_matching_history = None
        self.z_cut_points = None
        self.kmeans = None
        self.match_score = None
        self.indice = None
        self.done = False
        self.nb_mega_match = 0


    def step(self, action):
        # check if behind done
        if self.done: #self.user_item_history.sum() >= self.sampling_limit:
            print("You are calling step after it return done=True.\n"
                  "You should reset the environment.")

        assert len(action) <= self.action_size
        self.action = action

        # compute potential rewards
        potential_rewards = np.zeros((self.nb_users_men, self.nb_users_women))
        for man in range(self.nb_users_men):
            for woman in range(self.nb_users_women):
                if(self.user_match_history[man][woman] == 0):
                    potential_rewards[man][woman] = self._get_user_match(man,woman)

        #Let's compute the optimal number of good recommendation
        cost = -potential_rewards
        row, col = linear_sum_assignment(cost)
        optimal_reward = -cost[row, col].sum()

        # map couple as already recommended
        left_app = 0

        for p in action:
            self.user_match_history[p[0],p[1]] = 1

            #We won't recommend couples that left the app
            if(self._get_user_match(p[0],p[1])) == self.mega_score:

                left_app += 1
                self.nb_mega_match += 1
                self.user_match_history[p[0],:] = 1
                self.user_match_history[:,p[1]] = 1

            if (self._get_user_match(p[0],p[1])) == self.matching_score:
                self.user_matching_history[p[0],p[1]] = 1
        # compute reward R_t
        self.current_match = [self._get_user_match(p[0],p[1]) for p in action]
        self.reward = self.current_match

        #Compute the number of men and women starting using the left_app
        new_user_man = left_app
        new_user_woman = left_app

        #Manage entering and leaving people from the app
        #self.update_new_users(new_user_man, new_user_woman, index_left_app)
        self.replace_full_rec(self.user_match_history)


        # check if done
        if self.user_match_history.sum() == self.sampling_limit:
            self.done = True

        #Compute the possible recommendations based on the match history
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :] ==0).flatten()] for j in range(self.nb_users_men)]

        return self.reward, self.men_class, self.women_class, self.possible_recommendation, self.done, optimal_reward

    #Distribution of match probabilities
    #Each class has: a top class (50% match probability, 10% super-match probability)
    #                a second top class (35% match probability, 5% super-match probability)
    #                regular other classes (8% match probability, 2% super-match probability)
    def Proba(self,nb_classes):
        score = []
        top = self._rng.choice(nb_classes, nb_classes, replace=False)

        for i in range(nb_classes):
            match_score=[]
            alpa = [j for j in range(nb_classes)]
            alpa.remove(top[i])

            second = self._rng.choice(alpa,1)
            for j in range(nb_classes):
                if j==top[i]:
                    match_score.append([0.2,0.6])
                elif j == second:
                    match_score.append([0.35,0.85])
                else :
                    match_score.append([0.8,0.99])
            score.append(match_score)
        return score

    def reset(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self.action_size = min(self.nb_users_men, self.nb_users_women)
        #Make more than we need data
        self.X, self.y = make_blobs(n_samples=(self.nb_users_men + self.nb_users_women)*100, n_features=self.internal_embedding_size, centers=self.nb_classes, cluster_std=self.std, center_box=(-1.0, 1.0), shuffle=True, random_state=self._rng)
        self.indice = [i for i in range((self.nb_users_men+self.nb_users_women)*100)]
        #Delete the users indices that start coming in the app
        indice = np.random.choice(self.indice, self.nb_users_men+self.nb_users_women, replace=False)

        for el in indice:
          self.indice.remove(el)

        #Men and women features
        self.men_embedding = self.X[indice[0:self.nb_users_men]]
        self.women_embedding = self.X[indice[self.nb_users_men:self.nb_users_men+self.nb_users_women]]
        #Classes
        self.men_class = self.y[indice[0:self.nb_users_men]]
        self.women_class = self.y[indice[self.nb_users_men:self.nb_users_men+self.nb_users_women]]
        self.match_score = self.Proba(self.nb_classes)

        z_mean = self.men_mean.dot(self.women_mean)
        z_var = self.men_var.dot(self.women_var) + self.men_var.dot(np.square(self.women_mean)) + \
                self.women_var.dot(np.square(self.men_mean))
        z = norm(z_mean, np.sqrt(z_var))
        self.z_cut_points = z.ppf([0.5, 0.9]) # you can control the distribution of matches here.
        self.user_match_history = np.zeros((self.nb_users_men, self.nb_users_women))
        self.user_matching_history = np.zeros((self.nb_users_men, self.nb_users_women))
        self.done = False

        #Compute the possible recommendation based on the match history
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :] ==
            0).flatten()] for j in range(self.nb_users_men)]

        return self.men_class, self.women_class, self.possible_recommendation

    #Control rewards distribution according to the match type
    def _get_user_match(self, user1, user2):
        p = np.random.random()
        if p < self.match_score[self.men_class[user1]][self.women_class[user2]][0]:
            score = 0
        elif self.match_score[self.men_class[user1]][self.women_class[user2]][0]<=p < self.match_score[self.men_class[user1]][self.women_class[user2]][1]:
            score = self.matching_score
        else:
            score = self.mega_score
        return score

    #Return nb_users new users and their features
    def get_new_user(self, nb_users):
      indice = self._rng.choice(self.indice, nb_users, replace=False)

      for el in indice:
        self.indice.remove(el)

      X_user, y_user = self.X[indice[0:nb_users]], self.y[indice[0:nb_users]]
      return X_user, y_user

    #Get people who has been recommended to every one and make him leave the app
    #Replace the same number of leaving people by new users
    def indice_full_woman(self,user_match_history):
        nb_user_women = self.nb_users_women
        nb_user_men = self.nb_users_men
        indice_woman_full=[]
        for j in range(nb_user_women):
            if(user_match_history[:,j].sum() == nb_user_men):
                indice_woman_full.append(j)
        return indice_woman_full

    def indice_full_man(self,user_match_history):
        nb_user_women = self.nb_users_women
        nb_user_men = self.nb_users_men
        indice_man_full=[]
        for i in range(nb_user_men):

            if(user_match_history[i,:].sum() == nb_user_women):
                indice_man_full.append(i)
        return indice_man_full

    def replace_full_rec(self, user_match_history):
            nb_user_men = self.nb_users_men
            nb_user_women = self.nb_users_women
            #Check men users
            indice_man_full = self.indice_full_man(user_match_history)
            indice_woman_full = self.indice_full_woman(user_match_history)

            man_embedding, man_class = self.get_new_user(len(indice_man_full))
            woman_embedding, woman_class = self.get_new_user(len(indice_woman_full))
            #Delete previous match, features and classes
            self.men_embedding = np.delete(self.men_embedding, indice_man_full, 0)
            self.user_match_history = np.delete(self.user_match_history, indice_man_full, 0)
            self.user_matching_history = np.delete(self.user_matching_history, indice_man_full, 0)
            self.men_class = np.delete(self.men_class, indice_man_full)
            #Append new features, history and class
            self.men_embedding = np.append(self.men_embedding, man_embedding, axis=0)
            self.user_match_history = np.r_[self.user_match_history, np.zeros((len(indice_man_full),self.nb_users_women))]
            self.user_matching_history = np.r_[self.user_matching_history, np.zeros((len(indice_man_full),self.nb_users_women))]
            self.men_class = np.append(self.men_class, man_class, axis=0)

            #Check women users
            self.women_embedding = np.delete(self.women_embedding, indice_woman_full, 0)
            self.user_match_history = np.delete(self.user_match_history, indice_woman_full, 1)
            self.user_matching_history = np.delete(self.user_matching_history, indice_woman_full, 1)
            self.women_class = np.delete(self.women_class, indice_woman_full)
            #Append new features, history and class
            self.women_embedding = np.append(self.women_embedding, woman_embedding, axis=0)
            self.user_match_history = np.c_[self.user_match_history, np.zeros((self.nb_users_men,len(indice_woman_full)))]
            self.user_matching_history = np.c_[self.user_matching_history, np.zeros((self.nb_users_men,len(indice_woman_full)))]
            self.women_class = np.append(self.women_class, woman_class, axis=0)


            self.action_size = min(self.nb_users_men, self.nb_users_women)
            self.sampling_limit = self.nb_users_men * self.nb_users_women
