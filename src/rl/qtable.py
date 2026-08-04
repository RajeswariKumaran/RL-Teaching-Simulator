"""
qtable.py

Stores Q-values for state-action pairs.

Q(s, a) represents the estimated value of taking
action a while in state s.
"""

from collections import defaultdict

from src.actions import Action


class QTable:

    def __init__(self):
        """
        Create an empty Q-table.

        Each state will automatically get a Q-value
        of 0.0 for every possible action.
        """

        self.table = defaultdict(
            lambda: {
                action: 0.0
                for action in Action
            }
        )

    def get_q_value(self, state, action):
        """
        Return Q(state, action).
        """

        return self.table[state][action]

    def set_q_value(self, state, action, value):
        """
        Set Q(state, action) to a new value.
        """

        self.table[state][action] = value

    def get_state_values(self, state):
        """
        Return all Q-values for a state.
        """

        return self.table[state]

    def get_max_q(self, state):
        """
        Return the highest Q-value for a state.
        """

        return max(self.table[state].values())

    def get_best_action(self, state):
        """
        Return the action with the highest Q-value.
        """

        return max(
            self.table[state],
            key=self.table[state].get
        )