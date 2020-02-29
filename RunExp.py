import matplotlib.pyplot as plt

def run_exp(agent, env, nb_steps, env_seed):
    rewards = np.zeros(nb_steps)
    regrets = np.zeros(nb_steps)

    men_embedding, women_embedding, possible_recommendation = env.reset(env_seed)

    for i in range(nb_steps):
        # Select action from agent policy.
        recommendation = agent.act(men_embedding, women_embedding, possible_recommendation)

        # Play action in the environment and get reward.
        reward, men_embedding, women_embedding, possible_recommendation, done, optimal_reward = env.step(recommendation)

        # Update agent.
        agent.update(reward)
        #context = next_context

        # Save history.
        #context[i] = context
        rewards[i] = reward
        regrets[i] = optimal_reward - reward

    reward = rewards.sum()
    regret = np.sum(regrets)


    plt.plot(regrets.mean(axis=0), color='blue')
    plt.plot(np.quantile(regrets, 0.05,axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets, 0.95,axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret.mean()))
    plt.xlabel('steps')
    plt.ylabel('regret')
    plt.show()

    return {'reward': reward,
            'regret': regret,
            'rewards': rewards,
            'regrets': regrets,
            'cum_rewards': np.cumsum(rewards),
            'cum_regrets': np.cumsum(regrets),
            }
