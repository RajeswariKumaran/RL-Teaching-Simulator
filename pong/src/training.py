import torch
import torch.nn.functional as F
import numpy as np

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
    experiences, indices, weights = replay_buffer.sample(
        batch_size
    )

    states, actions, rewards, next_states, dones = zip(*experiences)

    # 3. Move everything to the selected device
    states = torch.tensor(
        np.array(states),
        dtype=torch.float32,
        device=device,
    )/255.0

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
    if torch.rand(1).item() < 0.005:
        print(
            "Reward distribution:",
            "negative =", (rewards < 0).sum().item(),
            "zero =", (rewards == 0).sum().item(),
            "positive =", (rewards > 0).sum().item(),
        )
    next_states = torch.tensor(
        np.array(next_states),
        dtype=torch.float32,
        device=device,
    )/255.0

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device,
    )
    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
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

        # Online network chooses the best action
        next_actions = model(next_states).argmax(
            dim=1,
            keepdim=True
        )

        # Target network evaluates that action
        next_q_values = target_model(next_states)

        max_next_q_values = next_q_values.gather(
            1,
            next_actions
        ).squeeze(1)

        # Bellman target
        target_q_values = rewards + (
            gamma
            * max_next_q_values
            * (1 - dones)
        )
        if torch.rand(1).item() < 0.001:
            print(
                "TRAIN DIAGNOSTIC:",
                "reward min/max =", rewards.min().item(), rewards.max().item(),
                "target min/max =", target_q_values.min().item(), target_q_values.max().item(),
                "current Q min/max =", current_q_values.min().item(), current_q_values.max().item(),
            )

    # 6. Compare the model's prediction with the Bellman target
    td_errors = (
        target_q_values - current_q_values
    )

    losses = F.smooth_l1_loss(
        current_q_values,
        target_q_values,
        reduction="none"
    )

    loss = (
        weights * losses
    ).mean()

    # 7. Update the neural network
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    new_priorities = (
        td_errors.detach()
        .abs()
        .cpu()
        .numpy()
    )

    replay_buffer.update_priorities(
        indices,
        new_priorities
    )

    return loss.item()