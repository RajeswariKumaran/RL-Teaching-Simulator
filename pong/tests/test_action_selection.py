import torch
import torch.nn as nn

from src.action_selection import select_action


class DummyModel(nn.Module):
    """
    A simple model that always returns the same Q-values.
    """

    def forward(self, state):
        return torch.tensor([[1.0, 2.0, 3.0]])


def test_select_action_with_zero_epsilon():

    model = DummyModel()
    state = torch.zeros((1, 4, 84, 84))

    action = select_action(
        model,
        state,
        epsilon=0.0
    )

    # Exploitation should choose the largest Q-value.
    # Q-values = [1, 2, 3], so action index 2 is best.
    assert action == 2


def test_select_action_with_full_epsilon():

    model = DummyModel()
    state = torch.zeros((1, 4, 84, 84))

    action = select_action(
        model,
        state,
        epsilon=1.0
    )

    # Exploration should always return a valid random action.
    assert action in [0, 1, 2]