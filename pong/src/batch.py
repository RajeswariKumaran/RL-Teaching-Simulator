import numpy as np
import torch


def prepare_batch(batch, device="cpu"):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.as_tensor(
        np.stack(states),
        dtype=torch.float32,
        device=device,
    ) / 255.0

    actions = torch.as_tensor(
        actions,
        dtype=torch.int64,
        device=device,
    )

    rewards = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.as_tensor(
        np.stack(next_states),
        dtype=torch.float32,
        device=device,
    ) / 255.0

    dones = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=device,
    )

    return states, actions, rewards, next_states, dones
