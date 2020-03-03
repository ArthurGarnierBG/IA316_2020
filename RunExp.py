#Import libs
import matplotlib.pyplot as plt
import numpy as np
import argparse
#Import agents
from Environment import TinderEnv
from Smart_Random_Agent import Smart_Random_Agent
from Pure_Random_Agent import Pure_Random_Agent
from Epsilon_Greedy_Agent import Epsilon_Greedy_Agent
from UCB_Agent import UCB_Agent


def run_exp(agent, env, nb_steps, env_seed):
    rewards = np.zeros(nb_steps)
    regrets = np.zeros(nb_steps)
    nb_user_men = np.zeros(nb_steps)
    nb_user_women = np.zeros(nb_steps)

    men_embedding, women_embedding, men_class, women_class, possible_recommendation = env.reset(env_seed)

    for i in range(nb_steps):
        #print("\nStep "+str(i))
        #print("Nb steps:"+str(nb_steps))
        #print(env.user_match_history)
        # Select action from agent policy.
        #print("\nPossible recommendations : "+str(possible_recommendation))
        #Smart random
        #recommendation = agent.act(men_class, women_class, possible_recommendation, env.user_match_history)
        #Random Pure
        #recommendation = agent.act(men_embedding, women_embedding, men_class, women_class, possible_recommendation)
        #Epsilon_Greedy_Agent
        recommendation = agent.act(men_class, women_class, possible_recommendation, env.user_match_history)

        #print("Agent recommendation : "+str(recommendation))
        #print(men_class)
        # Play action in the environment and get reward.
        rewards_list, men_embedding, women_embedding, men_class, women_class, possible_recommendation, done, optimal_reward = env.step(recommendation)
        #print("Env reward :"+str(reward))
        # Update agent. careful possible_recommendation of former state
        #agent.update(rewards_list)
        agent.update(rewards_list, recommendation, men_class, women_class)
        #context = next_context

        # Save history.
        tot_reward = np.array(rewards_list).sum()
        rewards[i] = tot_reward
        regrets[i] = optimal_reward - tot_reward
        nb_user_men[i] = env.nb_users_men
        nb_user_women[i] = env.nb_users_women

    reward = np.sum(rewards)
    regret = np.sum(regrets)
    #print("Rewards: "+str(rewards))

    #print("\nRewards at each iteration : "+str(rewards))
    #print("Regrets at each iteration : "+str(regrets))
    #print("Total reward : "+str(reward))
    #print("Total regret : "+str(regret))

    return {'reward': reward,
            'regret': regret,
            'rewards': rewards,
            'regrets': regrets,
            'cum_rewards': np.cumsum(rewards),
            'cum_regrets': np.cumsum(regrets),
            'nb_user_men': nb_user_men,
            'nb_user_women': nb_user_women
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=int, default=10, help="Number of experience to simulate")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of steps so simulate in every experience")
    args = vars(parser.parse_args())
    nb_exp = args["exp"]
    nb_steps = args["steps"]
    seed = 2020
    regret = np.zeros(nb_exp)
    regrets = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        #agent = Smart_Random_Agent(seed=seed, nb_classes=env.nb_classes)
        #agent = Epsilon_Greedy_Agent(seed=seed, epsilon=0.3, nb_classes=env.nb_classes)
        #agent = UCB_Agent(seed=seed, c=1, nb_classes=env.nb_classes)
        
        agent = Pure_Random_Agent(seed=seed)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret[i] = exp['regret']
        regrets[i] = exp['cum_regrets']
    print(env.match_score)

    
    #plt.xlabel('steps')
    #plt.ylabel('regret')
    #plt.show()

    regret2 = np.zeros(nb_exp)
    regrets2 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        agent = Smart_Random_Agent(seed=seed, nb_classes=env.nb_classes)
        #agent = Epsilon_Greedy_Agent(seed=seed, epsilon=0.3, nb_classes=env.nb_classes)
        #agent = UCB_Agent(seed=seed, c=1, nb_classes=env.nb_classes)
        #agent = Pure_Random_Agent(seed=seed)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret2[i] = exp['regret']
        regrets2[i] = exp['cum_regrets']
    print(env.match_score)

    #plt.xlabel('steps')
    #plt.ylabel('regret')
    #plt.show()

    regret3 = np.zeros(nb_exp)
    regrets3 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        #agent = Smart_Random_Agent(seed=seed, nb_classes=env.nb_classes)
        agent = Epsilon_Greedy_Agent(seed=seed, epsilon=0.3, nb_classes=env.nb_classes)
        #agent = UCB_Agent(seed=seed, c=1, nb_classes=env.nb_classes)
        #agent = Pure_Random_Agent(seed=seed)

        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret3[i] = exp['regret']
        regrets3[i] = exp['cum_regrets']
    print(env.match_score)
    print(agent._q)

    
    #plt.xlabel('steps')
    #plt.ylabel('regret')
    #plt.show()

    regret4 = np.zeros(nb_exp)
    regrets4 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        #agent = Smart_Random_Agent(seed=seed, nb_classes=env.nb_classes)
        #agent = Epsilon_Greedy_Agent(seed=seed, epsilon=0.3, nb_classes=env.nb_classes)
        agent = UCB_Agent(seed=seed, c=1, nb_classes=env.nb_classes)
        #agent = Pure_Random_Agent(seed=seed)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret4[i] = exp['regret']
        regrets4[i] = exp['cum_regrets']
    print(env.match_score)
    print(agent._q)


    plt.plot(regrets.mean(axis=0), color='blue')
    plt.plot(regrets2.mean(axis=0), color='green')
    plt.plot(regrets3.mean(axis=0), color='yellow')
    plt.plot(regrets4.mean(axis=0), color='red')
    plt.legend(['Pure','Smart','Epsilon','Ucb'])
    plt.plot(np.quantile(regrets, 0.05,axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets, 0.95,axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret.mean()))
    
    plt.plot(np.quantile(regrets2, 0.05,axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets2, 0.95,axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret.mean()))
    plt.plot(np.quantile(regrets3, 0.05,axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets3, 0.95,axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret3.mean()))
    
    
    plt.plot(np.quantile(regrets4, 0.05,axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets4, 0.95,axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret.mean()))
    plt.xlabel('steps')
    plt.ylabel('regret')
    plt.show()
