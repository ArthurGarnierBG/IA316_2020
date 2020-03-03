from scipy.optimize import linear_sum_assignment
import numpy as np


class QLearning_Greedy:
    """ Creates an epsilon-greedy policy based on a given Q-function and epsilon"""
    def __init__(self, nb_classes, alpha=0.5, discount_factor=0.1, epsilon=0.1, seed=None):
        self.epsilon = epsilon
        self.nb_classes = nb_classes
        self._Q = np.zeros((nb_classes, nb_classes))
        self.discount_factor = discount_factor
        self.alpha = alpha

    def act(self, men_class, women_class, possible_recommendation, user_match_history):
        #Sort men by match history
        possible_recommendation, index_men = self.get_men_order(user_match_history, possible_recommendation)
        men_class = men_class[index_men]
        women_index = [i for i in range(user_match_history.shape[1])]
        i = 0
        recommendation = []
        #Remove already recommended women from man recommendation
        for man_rec in possible_recommendation:
            diff = np.setdiff1d(man_rec, women_index, assume_unique=False)
            for element in diff:
                man_rec.remove(element)
            #Perform Epsilon Greedy Q Learning
            A = np.ones(self.nb_classes, dtype=float) * self.epsilon/self.nb_classes
            best_woman_class = np.argmax(self._Q[men_class[i]])
            A[best_woman_class] += (1.0 - self.epsilon)
            woman_class = np.random.choice(np.arange(len(A)), p=A)
            woman = self.get_matching_woman(woman_class, women_class, man_rec, user_match_history, women_index)
            #Remove recommended woman from this iteration pool
            women_index.remove(woman)
            recommendation.append((index_men[i], woman))
            i+=1
        return(recommendation)

    def update(self, rewards, recommended_pairs, men_class, women_class):
        states = [i for i in range(self.nb_classes)]
        state_women = [[] for i in range(self.nb_classes)]
        reward = np.zeros(self.nb_classes)
        i = 0
        #Record the recommended woman class by man class and the sum of rewards by man class
        for pair in recommended_pairs:
            reward[men_class[pair[0]]] += rewards[i]
            state_women[men_class[pair[0]]].append(women_class[pair[1]])
            i += 1
        #We perform the update per man class
        for state in states:
            best_next_action = np.argmax(self._Q[state])
            td_target = reward[state] + self.discount_factor * self._Q[state][best_next_action]
            for woman_class in state_women[state]:
                td_delta = td_target - self._Q[state][woman_class]
                self._Q[state][woman_class] += self.alpha * td_delta


    def get_number_woman_match(self, woman, user_match_history):
        return(user_match_history[:,woman].sum())

    def get_number_man_match(self, man, user_match_history):
        return(user_match_history[man,:].sum())

    #Give the order in which the agent process the matching
    def get_men_order(self, user_match_history, possible_recommendation):
        index_men = np.arange(len(possible_recommendation))
        match_nb  = [self.get_number_man_match(i, user_match_history) for i in range(user_match_history.shape[0])]
        sorted_index = np.argsort(match_nb)
        sorted_recommendation = [possible_recommendation[sorted_index[i]] for i in range(len(sorted_index))]
        return(sorted_recommendation, sorted_index)

    #Get a matching pair for a man whose recommendation class is class_rec
    #We select first the less recommended women
    def get_matching_woman(self, class_rec, women_class, man_pos_recommendation, user_match_history, women_index):
        #Women of class class_rec in the possible recommendations of man
        possible_women = []
        for woman in man_pos_recommendation:
            if(women_class[woman] == class_rec):
                possible_women.append(woman)

        #Let's choose a random woman among the equal min nb of matchs from the right class
        if(possible_women != []):
            #Compute nb of match for each woman
            nb_matches = [self.get_number_woman_match(woman, user_match_history) for woman in possible_women]
            sorted_index = np.argsort(nb_matches)
            min_nb_match = min(nb_matches)
            #Get all women whose nb of match is minimal and return a random one
            equal_women = []
            for i,woman in enumerate(possible_women):
                if(nb_matches[i] == min_nb_match):
                    equal_women.append(woman)

            chosen_woman = np.random.choice(equal_women)
            return chosen_woman

        #Is no one is in the good class, pick a random woman in the remaining women from the pool
        else:
            chosen_woman = np.random.choice(women_index)
            return chosen_woman
