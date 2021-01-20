import tensorflow as tf
import keras
import keras.layers as L
import numpy as np
import random

tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)


def get_network(state_dim, n_actions):
    """
    return model for training

    Args
        state_dim (int): dimension of state
        n_actions (int): num of actions

    Return
        network: keras.models.Sequential()
    """

    network = keras.models.Sequential()
    network.add(L.InputLayer(state_dim))
    network.add(L.Dense(50))
    network.add(L.Dense(n_actions))

    assert network.layers[-1].activation == keras.activations.linear,\
        "please make sure you predict q-values without nonlinearity"

    return network

# -------------------------------------------------------------------------


def get_action(state, network, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action
                    with highest Q(s,a)

    Args
        state:
        network: model for inference
        epsilon (option) : greedy parameter

    Returns
        chosen_action: list 選ばれたアクション
    """

    q_values = network.predict(state[None])[0]

    best_or_random = np.random.binomial(n=1, p=epsilon)
    best_action = np.argmax(q_values)

    if best_or_random == 0:
        chosen_action = best_action
    else:
        chosen_action = random.choice(list(range(len(q_values))))

    # return <epsilon-greedily selected action>
    return chosen_action

# -----------------------------------------------------------------------------


def generate_session(env, network, train_step, states_ph,
                     actions_ph, rewards_ph, next_states_ph,
                     is_done_ph, t_max=1000, epsilon=0, train=False):
    """
    play env with approximate q-learning agent and train it at the same time
    """
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        a = get_action(s, network, epsilon=epsilon)
        next_s, r, done, _ = env.step(a)

        if train:
            sess.run(train_step, {
                states_ph: [s], actions_ph: [a], rewards_ph: [r],
                next_states_ph: [next_s], is_done_ph: [done]
            })

        total_reward += r
        s = next_s
        if done:
            break

    return (total_reward, train_step)
