import tensorflow as tf
import keras
import gym
import gym.wrappers
import numpy as np
from approx_q_learning import get_network, generate_session


def train(save_video=True):

    env = gym.make("CartPole-v0").env
    env.reset()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape

    network = get_network(state_dim, n_actions)

    # Create placeholders for the <s, a, r, s'> tuple and a
    # special indicator for game end (is_done = True)
    states_ph = keras.backend.placeholder(
        dtype='float32', shape=(None,) + state_dim)
    actions_ph = keras.backend.placeholder(dtype='int32', shape=[None])
    rewards_ph = keras.backend.placeholder(dtype='float32', shape=[None])
    next_states_ph = keras.backend.placeholder(
        dtype='float32', shape=(None,) + state_dim)
    is_done_ph = keras.backend.placeholder(dtype='bool', shape=[None])

    # get q-values for all actions in current states
    predicted_qvalues = network(states_ph)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = tf.reduce_sum(
        predicted_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)

    gamma = 0.99

    # compute q-values for all actions in next states
    # predicted_next_qvalues = <apply network to get q-values
    # for next_states_ph>
    predicted_next_qvalues = network(next_states_ph)

    # compute V*(next_states) using predicted next q-values
    next_state_values = tf.reduce_max(predicted_next_qvalues,
                                      reduction_indices=[1])

    # compute "target q-values" for loss - it's what's inside square
    # parentheses in the above formula.
    target_qvalues_for_actions_temp = predicted_qvalues_for_actions - \
        (rewards_ph + (gamma*next_state_values))
    target_qvalues_for_actions = tf.reduce_sum(
        target_qvalues_for_actions_temp * tf.one_hot(actions_ph, n_actions),
        axis=1)

    # at the last state we shall use simplified formula: Q(s,a) = r(s,a)
    # since s' doesn't exist
    target_qvalues_for_actions = tf.where(
        is_done_ph, rewards_ph, target_qvalues_for_actions)

    # mean squared error loss to minimize
    loss = (predicted_qvalues_for_actions -
            tf.stop_gradient(target_qvalues_for_actions)) ** 2
    loss = tf.reduce_mean(loss)

    # training function that resembles agent.update(state, action,
    # reward, next_state) from tabular agent
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    assert tf.gradients(loss, [predicted_qvalues_for_actions])[
        0] is not None, "make sure you update q-values for chosen actions\
             and not just all actions"
    assert tf.gradients(loss, [predicted_next_qvalues])[
        0] is None, "make sure you don't propagate gradient w.r.t. Q_(s',a')"
    assert predicted_next_qvalues.shape.ndims == 2,\
        "make sure you predicted q-values for all actions in next state"
    assert next_state_values.shape.ndims == 1,\
        "make sure you computed V(s') as maximum over just\
         the actions axis and not all axes"
    assert target_qvalues_for_actions.shape.ndims == 1,\
        "there's something wrong with target q-values, they must be a vector"

    epsilon = 0.5

    for i in range(1000):
        results = [generate_session(env, network, train_step, states_ph,
                                    actions_ph, rewards_ph, next_states_ph,
                                    is_done_ph, epsilon=epsilon, train=True)
                   for _ in range(100)]

        session_rewards = np.array([reward for (reward, _) in results])
        train_step = results[-1][1]

        print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(
            i, np.mean(session_rewards), epsilon))

        epsilon *= 0.99
        assert epsilon >= 1e-4,\
            "Make sure epsilon is always nonzero during training"

        if np.mean(session_rewards) > 300:
            print("You Win!")
            break

    if save_video:
        env = gym.wrappers.Monitor(
            gym.make("CartPole-v0"), directory="videos", force=True)
        _ = [generate_session(env, network, train_step, states_ph,
                              actions_ph, rewards_ph, next_states_ph,
                              is_done_ph, epsilon=epsilon, train=True)
             for _ in range(100)]
        env.close()


if __name__ == '__main__':
    train()
