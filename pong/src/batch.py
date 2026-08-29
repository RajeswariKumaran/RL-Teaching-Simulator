import numpy as np
import torch


def prepare_batch(batch, device="cpu"):
    """
    Convert a list of replay-buffer experiences into PyTorch tensors.
    """

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(
        np.array(states),
        dtype=torch.float32,
        device=device
    )

    actions = torch.tensor(
        actions,
        dtype=torch.int64,
        device=device
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        np.array(next_states),
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device
    )

    return states, actions, rewards, next_states, dones