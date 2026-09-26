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
    if len(replay_buffer) < batch_size:
        return None

    # Sample experiences
    batch = replay_buffer.sample(batch_size)

    # Convert the sampled batch to tensors
    # and normalize image states to 0-1
    states, actions, rewards, next_states, dones = prepare_batch(
        batch,
        device=device,
    )

    # -------------------------------------------------
    # Current Q-value
    # -------------------------------------------------

    current_q_values = model(states).gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    # -------------------------------------------------
    # Double DQN target
    # -------------------------------------------------

    with torch.no_grad():

        # Online network chooses the best action
        next_actions = model(next_states).argmax(
            dim=1,
            keepdim=True,
        )

        # Target network evaluates that action
        next_q_values = target_model(next_states).gather(
            1,
            next_actions,
        ).squeeze(1)

        # Bellman target
        target_q_values = (
            rewards
            + gamma * next_q_values * (1.0 - dones)
        )

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    loss = F.smooth_l1_loss(
        current_q_values,
        target_q_values,
    )

    # -------------------------------------------------
    # Update online network
    # -------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    return loss.item()