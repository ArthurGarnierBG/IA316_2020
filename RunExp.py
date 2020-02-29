#Import libs
import matplotlib.pyplot as plt
import numpy as np
import argparse
#Import agents
from Environment import TinderEnv
from Smart_Random_Agent import Smart_Random_Agent
from Pure_Random_Agent import Pure_Random_Agent


def run_exp(agent, env, nb_steps, env_seed):
    rewards = np.zeros(nb_steps)
    regrets = np.zeros(nb_steps)
    nb_user_men = np.zeros(nb_steps)
    nb_user_women = np.zeros(nb_steps)

    men_embedding, women_embedding, possible_recommendation = env.reset(env_seed)

    for i in range(nb_steps):
        # Select action from agent policy.
        #print("\nPossible recommendations : "+str(possible_recommendation))
        recommendation = agent.act(men_embedding, women_embedding, possible_recommendation)
        #print("Agent recommendation : "+str(recommendation))
        # Play action in the environment and get reward.
        print("\nStep "+str(i))
        print(env.user_match_history)
        
        reward, men_embedding, women_embedding, possible_recommendation, done, optimal_reward = env.step(recommendation)
        #print("Env reward :"+str(reward))
        # Update agent. careful possible_recommendation of former state
        agent.update(reward)
        #context = next_context

        # Save history.
        rewards[i] = reward
        regrets[i] = optimal_reward - reward
        nb_user_men[i] = env.nb_users_men
        nb_user_women[i] = env.nb_users_women

    reward = rewards.sum()
    regret = np.sum(regrets)

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
    seed = 0
    regret = np.zeros(nb_exp)
    regrets = np.zeros((nb_exp, nb_steps))

    for i in range(nb_exp):
        env = TinderEnv(seed=seed)
        agent = Smart_Random_Agent(seed=seed)
        #agent = Pure_Random_Agent(seed=seed)
        exp = run_exp(agent, env, nb_steps, env_seed=seed)
        regret[i] = exp['regret']
        regrets[i] = exp['cum_regrets']
        #print("\nStep "+str(i))
        #print(exp['nb_user_men'])
        #print(exp['nb_user_women'])

    plt.plot(regrets.mean(axis=0), color='blue')
    plt.plot(np.quantile(regrets, 0.05,axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets, 0.95,axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret.mean()))
    plt.xlabel('steps')
    plt.ylabel('regret')
    plt.show()
