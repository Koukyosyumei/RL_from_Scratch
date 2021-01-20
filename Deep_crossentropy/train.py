import gym
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
from tqdm import tqdm
from deep_crossentropy import generate_session_mountain_car, select_elites_car


def train(n_sessions=50, percentile=90,
          hidden_layer_sizes=(30, 30), activation="tanh"):

    env = gym.make("MountainCar-v0").env
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    print("n_actions", n_actions)
    print("state_dim", state_dim)

    agent_car = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    # initialize agent to the dimension of state space and number of actions
    agent_car.partial_fit([env.reset()] * n_actions,
                          range(n_actions), range(n_actions))

    previous_states_batch = []
    previous_actions_batch = []
    previous_rewards_batch = []

    for i in tqdm(range(100)):
        # generate new sessions
        sessions = Parallel(
            n_jobs=-1)([delayed(generate_session_mountain_car)(env, agent_car)
                        for i in range(n_sessions)])

        states_batch, actions_batch, rewards_batch = map(list, zip(*sessions))

        current_states = states_batch + previous_states_batch
        current_actions = actions_batch + previous_actions_batch
        current_rewards = rewards_batch + previous_rewards_batch

        previous_states_batch = states_batch
        previous_actions_batch = actions_batch
        previous_rewards_batch = rewards_batch

        # <select elite actions just like before>
        elite_states, elite_actions = select_elites_car(
            current_states, current_actions, current_rewards, percentile)

        agent_car.partial_fit(elite_states, elite_actions)
        # <partial_fit agent to predict elite_actions(y) from elite_states(X)>

        mean_reward = np.mean(rewards_batch)
        threshold = np.percentile(rewards_batch, percentile)
        print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))

        if np.mean(rewards_batch) > -120:
            print("You Win! You may stop training now via KeyboardInterrupt.")
            break


if __name__ == '__main__':
    train()
