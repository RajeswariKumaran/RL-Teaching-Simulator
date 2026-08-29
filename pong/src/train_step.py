import torch

from src.batch import prepare_batch
import torch.nn.functional as F


def train_step(
    model,
    optimizer,
    replay_buffer,
    batch_size,
    gamma=0.99,
    device="cpu"
):
    """
    Perform one DQN training update.
    """

    # Make sure we have enough experiences
    if len(replay_buffer) < batch_size:
        return None

    # Sample experiences from the replay buffer
    batch = replay_buffer.sample(batch_size)

    # Convert the batch into PyTorch tensors
    states, actions, rewards, next_states, dones = prepare_batch(
        batch,
        device=device
    )

    # Get Q-values for the current states
    q_values = model(states)

    # Select Q(s, a) for the actions that were actually taken
    current_q_values = q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # Get Q-values for the next states
    with torch.no_grad():
        next_q_values = model(next_states)
        max_next_q_values = next_q_values.max(dim=1).values

    target_q_values = rewards + gamma * (
        1 - dones
        ) * max_next_q_values

    loss = F.mse_loss(
        current_q_values,
        target_q_values
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()