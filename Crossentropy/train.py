
import gym
import numpy as np

from tqdm import tqdm
from crossentropy import generate_session, select_elites,\
    update_policy, show_progress


def train(epochs=100, n_sessions=250, percentile=50, learning_rate=0.5):
    """ Summary line

        crossentropy method で Taxi-v2 を解く

        Args
            epochs           :
            n_sessions       : sample this many sessions
            percentile       : take this percent of session with highest reward
            learning_rate    : add this thing to all counts for stability
    """

    env = gym.make("Taxi-v2")
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # policy = an array to store action probabilities
    policy = np.full((n_states, n_actions), 1./n_actions)
    # reset policy just in case
    policy = np.ones([n_states, n_actions]) / n_actions

    log = []

    for i in tqdm(range(epochs)):

        # %time sessions = [ < generate a list of n_sessions new sessions > ]
        sessions = [generate_session(env, policy) for i in range(n_sessions)]

        states_batch, actions_batch, rewards_batch = zip(*sessions)
        # <select elite states/actions >
        elite_states, elite_actions = select_elites(
            states_batch, actions_batch, rewards_batch, percentile)

        # <compute new policy >
        new_policy = update_policy(
            elite_states, elite_actions, n_states, n_actions)
        policy = learning_rate*new_policy + (1-learning_rate)*policy

        # display results on chart
        show_progress(rewards_batch, log, percentile)


if __name__ == '__main__':
    train()
