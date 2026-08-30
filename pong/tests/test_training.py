import torch

from src.dqn import DQN
from src.replay_buffer import ReplayBuffer
from src.training import train_step


def create_experience():
    """Create one dummy Pong experience."""

    state = torch.randn(4, 84, 84).numpy()
    next_state = torch.randn(4, 84, 84).numpy()

    action = 1
    reward = 1.0
    done = False

    return state, action, reward, next_state, done


def test_train_step_returns_none_with_not_enough_experiences():
    model = DQN()
    replay_buffer = ReplayBuffer(capacity=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Add fewer experiences than the batch size
    for _ in range(2):
        replay_buffer.push(*create_experience())

    loss = train_step(
        model=model,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        batch_size=4,
        gamma=0.99,
        device=torch.device("cpu"),
    )

    assert loss is None


def test_train_step_updates_the_model():
    torch.manual_seed(42)

    model = DQN()
    replay_buffer = ReplayBuffer(capacity=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Add enough experiences for one batch
    for _ in range(4):
        replay_buffer.push(*create_experience())

    # Save the model parameters before training
    parameters_before = [
        parameter.clone().detach()
        for parameter in model.parameters()
    ]

    loss = train_step(
        model=model,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        batch_size=4,
        gamma=0.99,
        device=torch.device("cpu"),
    )

    # A training update should produce a loss
    assert loss is not None
    assert isinstance(loss, float)

    # At least one parameter should have changed
    parameters_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            model.parameters(),
        )
    )

    assert parameters_changed