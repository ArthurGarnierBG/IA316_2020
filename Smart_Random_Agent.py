from scipy.optimize import linear_sum_assignment
import numpy as np

class Smart_Random_Agent:
    """ Smart random agent. Each possible recommendation has a random weight, then Hungarian algorithm to determine the optimized assignment"""
    def __init__(self, nb_classes, seed=None):
        self._rng = np.random.RandomState(seed)
        self.nb_classes = nb_classes

    def act(self, men_class, women_class, possible_recommendation, user_match_history):
        possible_recommendation, index_men = self.get_men_order(user_match_history, possible_recommendation)
        women_index = [i for i in range(user_match_history.shape[1])]
        #print("Sorted men rec : "+str(possible_recommendation))
        #print("Index men: "+str(index_men))
        i = 0
        recommendation = []
        for man_rec in possible_recommendation:
            diff = np.setdiff1d(man_rec, women_index, assume_unique=False)
            #print("Dif: "+str(diff))
            for element in diff:
                man_rec.remove(element)
            #print("Woman index: "+str(women_index))
            class_rec = np.random.randint(self.nb_classes)
            #print("Recommended woman class: "+str(class_rec))
            #print("Women class : "+str(women_class))
            woman = self.get_matching_woman(class_rec, women_class, man_rec, user_match_history, women_index)
            #print("Woman chosen : "+str(woman))
            women_index.remove(woman)
            recommendation.append((index_men[i], woman))
            i+=1

        #print("Agent recommendation pair: "+str(recommendation))
        return(recommendation)

    def update(self, rewards, recommended_pairs, men_class, women_class):
        pass

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
    def get_matching_woman(self, class_rec, women_class, man_pos_recommendation, user_match_history, women_index):
        #Women of class class_rec in the possible recommendations of man
        possible_women = []
        for woman in man_pos_recommendation:
            if(women_class[woman] == class_rec):
                possible_women.append(woman)

        #print("Possible women:"+str(possible_women))
        #Let's choose a random woman among the equal min nb of matchs from the right class
        if(possible_women != []):
            #Compute nb of match for each woman
            nb_matches = [self.get_number_woman_match(woman, user_match_history) for woman in possible_women]
            #print("Matches of women "+str(nb_matches))
            sorted_index = np.argsort(nb_matches)
            #print("Sorted matches "+str(sorted_index))
            min_nb_match = min(nb_matches)
            #print("Min match :"+str(min_nb_match))
            #Get all women whose nb of match is minimal and return a random one
            equal_women = []
            for i,woman in enumerate(possible_women):
                if(nb_matches[i] == min_nb_match):
                    equal_women.append(woman)

            #print("Equal women: " +str(equal_women))
            chosen_woman = np.random.choice(equal_women)
            return chosen_woman

        #Is no one is in the good class, pick a random woman
        else:
            chosen_woman = np.random.choice(women_index)
            return chosen_woman
