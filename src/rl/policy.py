"""
policy.py

Defines policies used by the RL agent to select actions.
"""

import random
from dataclasses import dataclass
from enum import Enum

from src.actions import Action
from src.rl.qtable import QTable


class DecisionType(Enum):
    """Describes how an action was selected."""

    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"


@dataclass
class PolicyDecision:
    """
    Contains information about a policy decision.
    """

    action: Action
    decision_type: DecisionType
    random_value: float | None = None


class GreedyPolicy:

    def __init__(self, qtable: QTable):
        """
        Create a greedy policy.

        The policy always selects the action
        with the highest Q-value.
        """

        self.qtable = qtable

    def select_action(self, state):
        """
        Select the action with the highest Q-value.

        Greedy selection is always exploitation.
        """

        action = self.qtable.get_best_action(state)

        return PolicyDecision(
            action=action,
            decision_type=DecisionType.EXPLOITATION
        )


class EpsilonGreedyPolicy:

    def __init__(
        self,
        qtable: QTable,
        epsilon=0.1
    ):
        """
        Create an epsilon-greedy policy.

        Parameters:
            qtable: QTable containing the Q-values.
            epsilon: Probability of exploration.
        """

        self.qtable = qtable
        self.epsilon = epsilon

    def select_action(self, state):
        """
        Select an action using epsilon-greedy selection.

        With probability epsilon:
            Explore using a random action.

        Otherwise:
            Exploit using the best known action.
        """

        random_value = random.random()

        if random_value < self.epsilon:

            action = random.choice(list(Action))

            return PolicyDecision(
                action=action,
                decision_type=DecisionType.EXPLORATION,
                random_value=random_value
            )

        action = self.qtable.get_best_action(state)

        return PolicyDecision(
            action=action,
            decision_type=DecisionType.EXPLOITATION,
            random_value=random_value
        )