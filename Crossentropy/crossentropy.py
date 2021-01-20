from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def generate_session(env, policy, t_max=10**4):
    """ Sumaary line

    Play game until end or for t_max ticks.
    Args:
        env   : 実験の環境
        policy: an array of shape [n_states,n_actions] with action
                probabilities

    Returns:
        list of states, list of actions and sum of rewards

    """
    states, actions = [], []
    total_reward = 0.

    s = env.reset()

    for t in range(t_max):

        # sample action from policy
        a = np.random.choice(list(range(len(policy[s]))), p=policy[s])

        new_s, r, done, info = env.step(a)

        # Record state, action and add up reward to states,
        # actions and total_reward accordingly.
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """ Summary line

    Select states and actions from games that have rewards >= percentile

    Args
        states_batch  (list) : list of lists
                                of states, states_batch[session_i][t]
        actions_batch (list) : list of lists of
                                actions, actions_batch[session_i][t]
        rewards_batch (list) : list of rewards, rewards_batch[session_i]

    Returns
        elite_states  (1D lists) : 1D lists of states from elite sessions
        elite_actions (1D lists) : 1D lists of actions from elite sessions

    """

    # Compute minimum reward for elite session

    states_batch = np.array(states_batch)
    actions_batch = np.array(actions_batch)
    rewards_batch = np.array(rewards_batch)

    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = states_batch[rewards_batch >= reward_threshold]
    elite_actions = actions_batch[rewards_batch >= reward_threshold]

    elite_states = sum(elite_states, [])
    elite_actions = sum(elite_actions, [])

    return elite_states, elite_actions


def update_policy(elite_states, elite_actions, n_states, n_actions):
    """ Summary line
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    Args
        elite_states : 1D list of states from elite sessions
        elite_actions: 1D list of actions from elite sessions
        n_states     : statesの総数
        n_actions    : actionの総数

    Returns
        new_policy   : 2D - array ~ [states, actions]

    """

    elite_states = np.array(elite_states)
    elite_actions = np.array(elite_actions)

    new_policy = np.zeros([n_states, n_actions])

    for state in range(n_states):

        # normalize policy to get valid probabilities and handle 0/0 case.
        if state in elite_states:
            num_of_elite_states = np.count_nonzero(elite_states == state)
            pro4state = 1./num_of_elite_states

            for action in elite_actions[elite_states == state]:
                new_policy[state, action] += pro4state
        # In case you never visited a state, set probabilities for
        # all actions to 1./n_actions
        else:
            new_policy[state] = np.full(n_actions, 1./n_actions)

    return new_policy


def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
    """
    A convenience function that displays training progress.
    """

    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    clear_output(True)
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([np.percentile(rewards_batch, percentile)],
               [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    plt.show()
