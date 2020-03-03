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
from QLearning_Greedy import QLearning_Greedy


def run_exp(agent, env, nb_steps, env_seed):
    rewards = np.zeros(nb_steps)
    regrets = np.zeros(nb_steps)
    nb_user_men = np.zeros(nb_steps)
    nb_user_women = np.zeros(nb_steps)

    men_class, women_class, possible_recommendation = env.reset(env_seed)

    for i in range(nb_steps):
        #Agent recommendation pairs
        recommendation = agent.act(men_class, women_class, possible_recommendation, env.user_match_history)

        # Play action in the environment and get reward.
        rewards_list, men_class_next, women_class_next, possible_recommendation_next, done, optimal_reward = env.step(recommendation)

        # Update agent
        agent.update(rewards_list, recommendation, men_class, women_class)

        #Update state
        men_class = men_class_next
        women_class = women_class_next
        possible_recommendation = possible_recommendation_next


        # Save history.
        tot_reward = np.array(rewards_list).sum()
        rewards[i] = tot_reward
        regrets[i] = optimal_reward - tot_reward
        nb_user_men[i] = env.nb_users_men
        nb_user_women[i] = env.nb_users_women

    reward = np.sum(rewards)
    regret = np.sum(regrets)

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
        agent = Pure_Random_Agent(seed=seed)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret[i] = exp['regret']
        regrets[i] = exp['cum_regrets']

    regret2 = np.zeros(nb_exp)
    regrets2 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        agent = Smart_Random_Agent(seed=seed, nb_classes=env.nb_classes)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret2[i] = exp['regret']
        regrets2[i] = exp['cum_regrets']

    regret3 = np.zeros(nb_exp)
    regrets3 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        agent = Epsilon_Greedy_Agent(seed=seed, epsilon=0.2, nb_classes=env.nb_classes)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret3[i] = exp['regret']
        regrets3[i] = exp['cum_regrets']
    print("\n********** Epsilon Greedy policy **********")
    print(env.match_score)
    print(agent._q)

    regret4 = np.zeros(nb_exp)
    regrets4 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        agent = UCB_Agent(seed=seed, c=1, nb_classes=env.nb_classes)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret4[i] = exp['regret']
        regrets4[i] = exp['cum_regrets']
    print("\n********** UCB policy **********")
    print(env.match_score)
    print(agent._q)

    regret5 = np.zeros(nb_exp)
    regrets5 = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        agent = QLearning_Greedy(nb_classes=env.nb_classes, alpha=0.5, discount_factor=0.1, epsilon=0.1, seed=seed)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret5[i] = exp['regret']
        regrets5[i] = exp['cum_regrets']
    print("\n********** Q Learning policy **********")
    print(env.match_score)
    print(agent._Q)

    #Plot
    plt.plot(regrets.mean(axis=0), color='blue')
    plt.plot(regrets2.mean(axis=0), color='green')
    plt.plot(regrets3.mean(axis=0), color='yellow')
    plt.plot(regrets4.mean(axis=0), color='red')
    plt.plot(regrets5.mean(axis=0), color='black')

    plt.plot(np.quantile(regrets, 0.05,axis=0), color='blue', alpha=0.3)
    plt.plot(np.quantile(regrets, 0.95,axis=0), color='blue', alpha=0.3)

    plt.plot(np.quantile(regrets2, 0.05,axis=0), color='green', alpha=0.3)
    plt.plot(np.quantile(regrets2, 0.95,axis=0), color='green', alpha=0.3)

    plt.plot(np.quantile(regrets3, 0.05,axis=0), color='yellow', alpha=0.3)
    plt.plot(np.quantile(regrets3, 0.95,axis=0), color='yellow', alpha=0.3)

    plt.plot(np.quantile(regrets4, 0.05,axis=0), color='red', alpha=0.3)
    plt.plot(np.quantile(regrets4, 0.95,axis=0), color='red', alpha=0.3)

    plt.plot(np.quantile(regrets5, 0.05,axis=0), color='black', alpha=0.3)
    plt.plot(np.quantile(regrets5, 0.95,axis=0), color='black', alpha=0.3)

    plot_title = min([regret.mean(), regret2.mean(), regret3.mean(), regret4.mean(), regret5.mean()])
    plt.title('Mean regret: {:.2f}'.format(plot_title))
    plt.legend(['Pure','Smart','Epsilon','Ucb', 'QLearning'])
    plt.xlabel('steps')
    plt.ylabel('regret')
    plt.show()
