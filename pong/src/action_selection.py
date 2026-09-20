import random

import torch


def select_action(model, state, epsilon, num_actions=3):
    """Epsilon-greedy action selection."""

    if random.random() < epsilon:
        return random.randrange(num_actions)

    with torch.no_grad():
        q_values = model(state)

    return torch.argmax(q_values, dim=1).item()
