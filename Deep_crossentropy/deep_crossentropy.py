import numpy as np


def generate_session_mountain_car(env, agent, t_max=10000):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    states, actions = [], []
    total_reward = 0

    s = env.reset()

    for t in range(t_max):

        # use agent to predict a vector of action probabilities for state :s:
        probs = agent.predict_proba([s])  #
        probs = probs.reshape(-1)

        # use the probabilities you predicted to pick an action
        # sample proportionally to the probabilities
        a = np.random.choice(list(range(len(probs))), p=probs)  # <YOUR CODE>
        new_s, r, done, info = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


def select_elites_car(states_batch, actions_batch,
                      rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists
              of states and respective actions from elite sessions

    """

    states_batch_ = np.array(states_batch)
    actions_batch_ = np.array(actions_batch)
    rewards_batch_ = np.array(rewards_batch)

    reward_threshold = np.percentile(rewards_batch_, percentile)

    elite_states = states_batch_[rewards_batch_ >= reward_threshold]
    elite_actions = actions_batch_[rewards_batch_ >= reward_threshold]

    elite_states = sum(elite_states, [])
    elite_actions = sum(elite_actions, [])

    return elite_states, elite_actions
