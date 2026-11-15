import numpy as np
import torch

from src.dqn import DQN
from src.replay_buffer import ReplayBuffer
from src.train_step import train_step


def test_train_step_updates_model():

    # Create model and optimizer
    model = DQN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=10)

    # Add enough experiences for a batch
    for i in range(5):

        state = np.random.rand(
            4, 84, 84
        ).astype(np.float32)

        next_state = np.random.rand(
            4, 84, 84
        ).astype(np.float32)

        replay_buffer.push(
            state,
            i % 3,
            float(i),
            next_state,
            False
        )

    # Perform one learning update
    loss = train_step(
        model,
        optimizer,
        replay_buffer,
        batch_size=3
    )

    # The training step should produce a numerical loss
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0