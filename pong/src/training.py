import torch
import torch.nn.functional as F

from src.batch import prepare_batch


def train_step(
    model,
    replay_buffer=None,
    target_model=None,
    optimizer=None,
    batch_size=32,
    gamma=0.99,
    device="cpu",
    *args,
):
    """
    Perform one DQN update using the target network.

    The keyword-based signature is the canonical API used by the Pong
    trainer. A small backwards-compatibility path also accepts the older
    positional form:

        train_step(model, optimizer, replay_buffer, batch_size,
                   gamma=0.99, device="cpu")

    In that legacy form, the online network is used as the target because
    no separate target network was supplied.
    """

    # Backwards compatibility with the old train_step.py API.
    if isinstance(replay_buffer, torch.optim.Optimizer):
        old_optimizer = replay_buffer
        old_replay_buffer = target_model

        # Old signature was:
        # train_step(model, optimizer, replay_buffer, batch_size,
        #            gamma=0.99, device="cpu")
        if isinstance(optimizer, int):
            old_batch_size = optimizer
            old_gamma = batch_size
            old_device = gamma
        else:
            old_batch_size = batch_size
            old_gamma = gamma
            old_device = device

        replay_buffer = old_replay_buffer
        optimizer = old_optimizer
        target_model = model
        batch_size = old_batch_size

        if isinstance(old_gamma, (int, float)):
            gamma = old_gamma
        if isinstance(old_device, (torch.device, str)):
            device = old_device

        if args:
            if len(args) >= 1:
                gamma = args[0]
            if len(args) >= 2:
                device = args[1]

    if replay_buffer is None or optimizer is None:
        raise TypeError(
            "train_step requires model, replay_buffer, optimizer, "
            "batch_size, gamma, and device."
        )

    if target_model is None:
        target_model = model

    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)

    states, actions, rewards, next_states, dones = prepare_batch(
        batch,
        device=device,
    )

    current_q_values = model(states).gather(
        1,
        actions.unsqueeze(1),
    ).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(next_states).max(dim=1).values
        target_q_values = rewards + gamma * next_q_values * (1.0 - dones)

    loss = F.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()
