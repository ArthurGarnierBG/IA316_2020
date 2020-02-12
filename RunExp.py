def run_exp(agent, env, nb_steps, env_seed):
    rewards = np.zeros(nb_steps)
    regrets = np.zeros(nb_steps)
    actions = np.zeros(nb_steps)
    context = env.reset(env_seed)
    rating_matrix = np.zeros((env.nb_users, env.nb_items))
    for i in range(nb_steps):
        # Select action from agent policy.
        action = agent.act(context)
        
        # Play action in the environment and get reward.
        reward, next_context, done, optimal_return = env.step(action)
        
        # Update history
        user = context[0][0]
        item = context[action][1]
        rating = reward
        rating_matrix[user, item] = rating
        
        # Update agent.
        agent.update(context, action, reward)
        context = next_context
        
        # Save history.
        #context[i] = context
        rewards[i] = reward
        actions[i] = action
        regrets[i] = optimal_return - reward

    reward = rewards.sum()
    regret = np.sum(regrets)
    return {'reward': reward, 
            'regret': regret,
            'rewards': rewards,
            'regrets': regrets,
            'actions': actions,
            'cum_rewards': np.cumsum(rewards), 
            'cum_regrets': np.cumsum(regrets),
            'rating_matrix': rating_matrix
            }


