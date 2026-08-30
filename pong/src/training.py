import torch
import torch.nn.functional as F


def train_step(
    model,
    replay_buffer,
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

    states, actions, rewards, next_states, dones = zip(*experiences)

    # 3. Move everything to the selected device
    states = torch.tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long,
        device=device,
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32,
        device=device,
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device,
    )

    # 4. Get the Q-values predicted by the model
    q_values = model(states)

    # Select the Q-value corresponding to the action
    # actually taken in each experience
    current_q_values = q_values.gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    # 5. Calculate the Q-values for the next states
    with torch.no_grad():
        next_q_values = model(next_states)

        max_next_q_values = next_q_values.max(
            dim=1
        ).values

        # Bellman target
        target_q_values = rewards + (
            gamma
            * max_next_q_values
            * (1 - dones)
        )

    # 6. Compare the model's prediction with the Bellman target
    loss = F.mse_loss(
        current_q_values,
        target_q_values,
    )

    # 7. Update the neural network
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()