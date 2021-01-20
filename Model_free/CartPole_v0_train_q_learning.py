from qlearning import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from gym.core import ObservationWrapper
from IPython.display import clear_output

def play_and_train(env,agent,t_max=10**4):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        # get agent to pick action given state s.
        #a = <YOUR CODE>
        a = agent.get_action(s)

        next_s, r, done, _ = env.step(a)

        # train (update) agent for state s
        #<YOUR CODE HERE>
        agent.update(s, a, r, next_s, t, adaptive=False)

        s = next_s
        total_reward +=r
        if done:
            break

    return total_reward

def get_custom_Binarizer(n_digits, slide):


    class Binarizer(ObservationWrapper):

        def observation(self, state):

            #state = <round state to some amount digits.>
            #hint: you can do that with round(x,n_digits)
            #you will need to pick a different n_digits for each dimension
            #n_digits = [0, 1, 1, 1]
            #n_digits = n_digits
            #state = [round(s*1.5, n_digit)/1.5 for s, n_digit in zip(state, n_digits)]
            state = [round(s*slide, n_digit)/slide for s, n_digit in zip(state, n_digits)]
            #state = [round(s, n_digit) for s, n_digit in zip(state, n_digits)]

            return tuple(state)

    return Binarizer

def train():
    Binarizer = get_custom_Binarizer([0,1,1,1], 1.5)
    env = Binarizer(gym.make("CartPole-v0"))
    n_actions = env.action_space.n
    agent = QLearningAgent(alpha=0.4, epsilon=0.1, discount=0.99,
                       get_legal_actions=lambda s: range(n_actions))

    rewards = []
    for i in tqdm(range(10000)):
        rewards.append(play_and_train(env,agent))

        #OPTIONAL YOUR CODE: adjust epsilon
        if i %100 ==0:
            clear_output(True)
            print('eps =', agent.epsilon, 'mean reward =', np.mean(rewards[-10:]))
            plt.plot(rewards)
            plt.show()

if __name__ == '__main__':
    train()
