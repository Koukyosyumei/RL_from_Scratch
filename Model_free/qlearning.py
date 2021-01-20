from collections import defaultdict
import random
import numpy as np

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """ Summary line
        Q-Learning Agent
        based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

        Attribute
          - self.epsilon (floata) : exploration prob
          - self.alpha (float)    : learning rate
          - self.discount (float) : discount rate aka gamma

        Functions I use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value

        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def adaptive_rate(self, t):
        return 1.1 - max([0.1, min(1, np.log((t)/25))])

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    # ---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """ Summary line
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.

        Args
            state :

        Returns
            value :

        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        q_values = np.array([self.get_qvalue(state, action) for action in possible_actions])
        value    = np.max(q_values)

        return value

    def update(self, state, action, reward, next_state, t, adaptive=False):
        """ Summary line
           Q_value を更新する
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        if adaptive:
            learning_rate = self.alpha * self.adaptive_rate(t)
        else:
            learning_rate = self.alpha

        q_s_a = (1 - learning_rate) * self.get_qvalue(state, action) + learning_rate * (reward + gamma * self.get_value(next_state))

        self.set_qvalue(state, action, q_s_a)


    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #<YOUR CODE HERE>
        q_values = np.array([self.get_qvalue(state, action) for action in possible_actions])
        best_index    = np.argmax(q_values)
        best_action   = possible_actions[best_index]

        return best_action

    def get_action(self, state):
        """ Summaru line
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.getPolicy).

        Args
            state          :

        Returns
            chosen_action  :

        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        best_or_random = np.random.binomial(n=1, p=epsilon)

        if best_or_random == 0:
            chosen_action = self.get_best_action(state)
        else:
            chosen_action = random.choice(possible_actions)

        return chosen_action
