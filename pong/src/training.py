import torch
import torch.nn.functional as F

from src.batch import prepare_batch


def train_step(
    model,
    replay_buffer,
    target_model,
    optimizer,
    batch_size,
    gamma,
    device,
):
    """
    Perform one DQN training update.

    Returns the loss, or None if there are not enough experiences yet.
    """

    # 1. We cannot train until we have enough experiences for a batch
    if len(replay_buffer) < batch_size:
        return None

    # 2. Sample a random batch from the replay buffer
    experiences = replay_buffer.sample(batch_size)

    # 3. Convert experiences into tensors
    states, actions, rewards, next_states, dones = prepare_batch(
        experiences,
        device=device,
    )

    # 4. Q-values for the actions actually taken
    current_q_values = model(states).gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # 5. Bellman target
    with torch.no_grad():
        next_q_values = target_model(next_states).max(dim=1).values

        target_q_values = (
            rewards
            + gamma * next_q_values * (1 - dones)
        )

    # 6. Vanilla DQN uses MSE loss
    loss = F.mse_loss(
        current_q_values,
        target_q_values,
    )

    # 7. Update the online network
    optimizer.zero_grad()
    loss.backward()

    # 8. Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    return loss.item()