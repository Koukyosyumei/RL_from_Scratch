from qlearning import QLearningAgent
import numpy as np


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.

    """

    def get_value(self, state):
        """ Summary line

        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Args
            state :

        Returns
            state_value

        """

        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        q_values = np.array([self.get_qvalue(state, action)
                            for action in possible_actions])

        state_value = np.sum(q_values / len(possible_actions))

        return state_value
