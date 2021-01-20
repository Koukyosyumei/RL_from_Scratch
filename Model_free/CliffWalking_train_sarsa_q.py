import gym
import gym.envs.toy_text
from qlearning import QLearningAgent
from sarsa import EVSarsaAgent
from IPython.display import clear_output
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import time

def play_and_train(env,agent,t_max=10**4):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)

        next_s,r,done,_ = env.step(a)
        agent.update(s, a, r, next_s, t)

        s = next_s
        total_reward +=r
        if done:break

    return total_reward

def draw_policy(env, agent):
    """ Prints CliffWalkingEnv policy with arrows. Hard-coded. """
    n_rows, n_cols = env._cliff.shape

    actions = '^>v<'

    for yi in range(n_rows):
        for xi in range(n_cols):
            if env._cliff[yi, xi]:
                print(" C ", end='')
            elif (yi * n_cols + xi) == env.start_state_index:
                print(" X ", end='')
            elif (yi * n_cols + xi) == n_rows * n_cols - 1:
                print(" T ", end='')
            else:
                print(" %s " % actions[agent.get_best_action(yi * n_cols + xi)], end='')
        print()


def train():
    env = gym.envs.toy_text.CliffWalkingEnv()
    n_actions = env.action_space.n
    print(env.__doc__)
    print(" ")
    print("Our cliffworld has one difference from what's on the image: there is no wall.")
    print(" ")
    print("Agent can choose to go as close to the cliff as it wishes. x:start, T:exit, C:cliff, o: flat ground")
    env.render()
    time.sleep(5)

    agent_sarsa = EVSarsaAgent(alpha=0.25, epsilon=0.1, discount=0.99,
                       get_legal_actions = lambda s: range(n_actions))

    agent_ql = QLearningAgent(alpha=0.25, epsilon=0.1, discount=0.99,
                           get_legal_actions = lambda s: range(n_actions))


    moving_average = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values

    rewards_sarsa, rewards_ql = [], []

    for i in range(5000):
        rewards_sarsa.append(play_and_train(env, agent_sarsa))
        rewards_ql.append(play_and_train(env, agent_ql))
        # Note: agent.epsilon stays constant

        if i %100 ==0:
            clear_output(True)
            print('EVSARSA mean reward =', np.mean(rewards_sarsa[-100:]))
            print('QLEARNING mean reward =', np.mean(rewards_ql[-100:]))
            plt.title("epsilon = %s" % agent_ql.epsilon)
            plt.plot(moving_average(rewards_sarsa), label='ev_sarsa')
            plt.plot(moving_average(rewards_ql), label='qlearning')
            plt.grid()
            plt.legend()
            plt.ylim(-500, 0)
            plt.show()

    print("Q-Learning")
    draw_policy(env, agent_ql)

    print("SARSA")
    draw_policy(env, agent_sarsa)
