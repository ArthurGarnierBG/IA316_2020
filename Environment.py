import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs

class TinderEnv:

    def __init__(self,
                nb_users_men=10,
                nb_users_women=50,
                nb_classes = 4,
                internal_embedding_size=10,
                std = 5.0,
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
        self.men_embedding = None
        self.men_class = None
        self.women_embedding = None
        self.women_class = None
        #X?
        self.X = None
        #yy?
        self.y = None
        self.user_match_history = None
        self.z_cut_points = None
        self.kmeans = None
        self.match_score = None
        self.indice = None
        self.done = False


    def step(self, action):
        # check if behind done
        if self.done: #self.user_item_history.sum() >= self.sampling_limit:
            print("You are calling step after it return done=True.\n"
                  "You should reset the environment.")

        assert len(action) <= self.action_size
        self.action = action

        #print("Number user men : "+str(self.nb_users_men))
        #print("Number user women : "+str(self.nb_users_women))

        # compute potential rewards
        #potential_rewards = np.array([[self._get_user_match(j,i) for i in np.argwhere(self.user_match_history[j, :] == 0).flatten()] for j in range(self.nb_users_men)])
        potential_rewards = np.zeros((self.nb_users_men, self.nb_users_women))
        for man in range(self.nb_users_men):
            for woman in range(self.nb_users_women):
                if(self.user_match_history[man][woman] == 0):
                    potential_rewards[man][woman] = self._get_user_match(man,woman)

        #print("Potential rewards :"+str(potential_rewards))

        #Let's compute the optimal number of good recommendation
        cost = -potential_rewards
        row, col = linear_sum_assignment(cost)
        optimal_reward = -cost[row, col].sum()
        #optimal_reward = [self._get_user_match(row[i],col[i]) for i in range(len(col))]
        #print("Optimal reward :"+str(optimal_reward))

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

        #print("Couples left app: "+str(left_app))
        # compute reward R_t
        self.current_match = [self._get_user_match(p[0],p[1]) for p in action]
        #print("Current match : "+str(self.current_match))
        #self.reward = np.sum(self.current_match)
        self.reward = self.current_match

        #Compute the number of men and women starting using the left_app
        #new_user_man = np.random.randint(left_app+2)
        new_user_man = left_app
        #print("New user men: "+str(new_user_man))
        #new_user_woman = np.random.randint(left_app+2)
        new_user_woman = left_app
        #print("New user women: "+str(new_user_woman))

        #print("User match history before replacement: "+str(self.user_match_history))
        #print("Man class before replacement : "+str(self.men_class))
        #print("Woman class before replacement : "+str(self.women_class))

        self.update_new_users(new_user_man, new_user_woman, index_left_app)
        self.replace_full_rec(self.user_match_history)

        #print("Man class after replacement : "+str(self.men_class))
        #print("Woman class after replacement : "+str(self.women_class))
        #print("User match history after replacement: "+str(self.user_match_history))

        # check if done
        if self.user_match_history.sum() == self.sampling_limit:
            self.done = True

        #Compute the possible recommendations based on the match history
        self.possible_recommendation = [[i for i in np.argwhere(self.user_match_history[j, :] ==0).flatten()] for j in range(self.nb_users_men)]

        return self.reward, self.men_embedding, self.women_embedding, self.men_class, self.women_class, self.possible_recommendation, self.done, optimal_reward

    #Que fait cette fonction?
    def Proba(self,nb_classes):
        score = []
        top = np.random.choice(nb_classes,nb_classes,replace=False)
        for i in range(nb_classes):
            match_score=[]
            for j in range(nb_classes):
                if j==top[i]:
                    match_score.append([0.4,0.9])
                else :
                    match_score.append([0.9,0.96])
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
        #self.women_embedding = self.get_new_user_women(self.nb_users_women)
        #self.kmeans = KMeans(n_clusters=self.nb_classes, random_state=self._rng).fit(np.concatenate([self.men_embedding,self.women_embedding]))
        #kmeans.labels_)
        #kmeans.predict([[0, 0], [12, 3]]))
        #kmeans.cluster_centers_
        #kmeans.inertia_

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

        return self.men_embedding, self.women_embedding, self.men_class, self.women_class, self.possible_recommendation

    def _get_user_match(self, user1, user2):
        p = np.random.random()
        if p < self.match_score[self.men_class[user1]][self.women_class[user2]][0]:
            score = 0
        elif self.match_score[self.men_class[user1]][self.women_class[user2]][0]<=p < self.match_score[self.men_class[user1]][self.women_class[user2]][1]:
            score = 1
        else:
            score = 4
        #real_score = self.men_embedding[user1].dot(self.women_embedding[user2])
        #print("real score: "+str(real_score))
        #match_score = np.searchsorted(self.z_cut_points, real_score)
        #print("Match score: "+str(match_score))
        #match_score = match_score**2
        return score

    #Return features for new users
    #def get_new_user_men(self, nb_users_men):
    #    return self._rng.normal(loc=self.men_mean, scale=self.men_var, size=(nb_users_men, self.internal_embedding_size))

    #def get_new_user_women(self, nb_users_women):
    #    return self._rng.normal(loc=self.men_mean, scale=self.men_var, size=(nb_users_women, self.internal_embedding_size))
    def get_new_user(self, nb_users):
      indice = np.random.choice(self.indice, nb_users, replace=False)

      for el in indice:
        self.indice.remove(el)

      X_user, y_user = self.X[indice[0:nb_users]], self.y[indice[0:nb_users]]
      return X_user, y_user


    #Update embeddings and user_match_history
    def update_new_users(self, new_user_man, new_user_woman, index_left_couple):
        #Compute indices of men and women leaving the app
        if(index_left_couple != []):
            left_man_index = [index_left_couple[i][0] for i in range(len(index_left_couple))]
            left_woman_index = [index_left_couple[i][1] for i in range(len(index_left_couple))]
            #Delete left users from embeddings and history
            self.user_match_history = np.delete(self.user_match_history, left_man_index, 0)
            self.user_match_history = np.delete(self.user_match_history, left_woman_index, 1)
            self.men_embedding = np.delete(self.men_embedding, left_man_index, 0)
            self.women_embedding = np.delete(self.women_embedding, left_woman_index, 0)
            self.men_class = np.delete(self.men_class, left_man_index)
            self.women_class = np.delete(self.women_class, left_woman_index)
            #Update nb of men and women
            self.nb_users_men -= len(index_left_couple)
            self.nb_users_women -= len(index_left_couple)

        if(new_user_man > 0):
            man_embedding, man_class = self.get_new_user(new_user_man)
            #print("New man class : "+str(man_class))
            self.men_class = np.append(self.men_class, man_class, axis=0)
            self.men_embedding = np.append(self.men_embedding, man_embedding, axis=0)
            self.user_match_history = np.append(self.user_match_history, [np.zeros(self.nb_users_women)]*new_user_man, axis=0)
            self.nb_users_men += new_user_man

        if(new_user_woman > 0):
            woman_embedding, woman_class = self.get_new_user(new_user_woman)
            #print("New woman class : "+str(woman_class))
            self.women_class = np.append(self.women_class, woman_class, axis=0)
            self.women_embedding = np.append(self.women_embedding, woman_embedding, axis=0)
            self.user_match_history = np.append(self.user_match_history, [np.zeros(new_user_woman)]*self.nb_users_men, axis=1)
            self.nb_users_women += new_user_woman

        self.action_size = min(self.nb_users_men, self.nb_users_women)
        self.sampling_limit = self.nb_users_men * self.nb_users_women


    def replace_full_rec(self, user_match_history):
        nb_user_men = self.nb_users_men
        nb_user_women = self.nb_users_women

        for i in range(nb_user_men):
            if(user_match_history[i,:].sum() == nb_user_women):
                man_embedding, man_class = self.get_new_user(1)
                #print("New man class : "+str(man_class))
                #Delete previous match, features and classes
                self.men_embedding = np.delete(self.men_embedding, i, 0)
                self.user_match_history = np.delete(self.user_match_history, i, 0)
                self.men_class = np.delete(self.men_class, i)
                #Append new features, history and class
                self.men_embedding = np.append(self.men_embedding, man_embedding, axis=0)
                self.user_match_history = np.r_[self.user_match_history, np.zeros((1,self.nb_users_women))]
                self.men_class = np.append(self.men_class, man_class, axis=0)


        for j in range(nb_user_women):
            if(user_match_history[:,j].sum() == nb_user_men):
                woman_embedding, woman_class = self.get_new_user(1)
                #print("New woman class : "+str(woman_class))
                #Delete previous match, features and classes
                self.women_embedding = np.delete(self.women_embedding, j, 0)
                self.user_match_history = np.delete(self.user_match_history, j, 1)
                self.women_class = np.delete(self.women_class, j)
                #Append new features, history and class
                self.women_embedding = np.append(self.women_embedding, woman_embedding, axis=0)
                self.user_match_history = np.c_[self.user_match_history, np.zeros(self.nb_users_men)]
                self.women_class = np.append(self.women_class, woman_class, axis=0)

        self.action_size = min(self.nb_users_men, self.nb_users_women)
        self.sampling_limit = self.nb_users_men * self.nb_users_women



if __name__ == "__main__":
    env = TinderEnv()
    men_embedding,women_embedding,men_class,women_class ,possible_recommendation = env.reset(seed=2020)
    #print(np.array(men_embedding).shape)
    #print(np.array(women_embedding).shape)
    #print(np.array(env.user_match_history).shape)
    #print(np.array(possible_recommendation).shape)
    #print("Men embedding:"+str(men_embedding))
    #print("Woman embedding:"+str(women_embedding))
    #print(np.array(env.user_match_history))
    #print(possible_recommendation)
    for step in range(10):

      recommendation = agent.act(men_embedding,women_embedding,men_class,women_class,possible_recommendation)

      reward,men_embedding, women_embedding,men_class,women_class ,possible_recommendation, done, optimal_reward = env.step(recommendation)

      print("possible recommendation : "+str(possible_recommendation))
      print(step)
      print('reward: ', reward)
