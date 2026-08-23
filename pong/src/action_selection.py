import random

import torch


def select_action(model, state, epsilon, num_actions=3):
    """
    Select an action using epsilon-greedy exploration.
    """

    # Exploration: choose a random action
    if random.random() < epsilon:
        return random.randrange(num_actions)

    # Exploitation: choose the action with the highest Q-value
    with torch.no_grad():
        q_values = model(state)

    return torch.argmax(q_values, dim=1).item()