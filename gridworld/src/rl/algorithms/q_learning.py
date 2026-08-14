"""
q_learning.py

Implements the tabular Q-Learning algorithm.
"""

from src.rl.qtable import QTable


class QLearning:

    def __init__(self, qtable, learning_rate=0.1, gamma=0.9):
        """
        Create a Q-Learning agent.

        Parameters:
            qtable: QTable used to store Q-values
            learning_rate: How quickly new information changes Q-values
            gamma: How much future rewards are valued
        """

        self.qtable = qtable
        self.learning_rate = learning_rate
        self.gamma = gamma

    def update(self, state, action, reward, next_state, done):
        """
        Perform one Q-Learning update.

        Returns:
            new_q_value
        """

        # Current Q-value: Q(s, a)
        current_q = self.qtable.get_q_value(
            state,
            action
        )

        # Find the best Q-value in the next state
        max_next_q = self.qtable.get_max_q(
            next_state
        )

        # If the episode is finished, there is no future reward
        if done:
            max_next_q = 0.0

        # Calculate the target
        target = reward + self.gamma * max_next_q

        # Calculate the learning error
        error = target - current_q

        # Calculate the new Q-value
        new_q_value = (
            current_q
            + self.learning_rate * error
        )

        # Store the new Q-value
        self.qtable.set_q_value(
            state,
            action,
            new_q_value
        )

        return new_q_value