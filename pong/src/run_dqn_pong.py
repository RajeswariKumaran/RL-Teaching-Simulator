import torch
from collections import deque

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action
from src.replay_buffer import ReplayBuffer
from src.training import train_step
from src.target_network import update_target_network


def main():

    env = PongEnvironment()

    state_manager = PongState()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Using device:", device)

    model = DQN().to(device)
    print("Model device:", next(model.parameters()).device)

    target_model = DQN().to(device)
    update_target_network(model, target_model)
    target_model.eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    replay_buffer = ReplayBuffer(
        capacity=50_000
    )

    batch_size = 32

    training_start = 10_000
    train_frequency = 4
    target_update_frequency = 1_000

    gamma = 0.99

    num_episodes = 1
    max_steps_per_episode = 1_000

    total_training_timesteps = 1_000_000

    # ---------------------------------------------------------
    # Epsilon-greedy exploration
    # ---------------------------------------------------------
    epsilon_start = 1.0
    epsilon_end = 0.01

    # Decay epsilon during the first 10% of the
    # planned training timesteps.
    # total_training_timesteps = (
    #     num_episodes * max_steps_per_episode
    # )

    epsilon_decay_steps = int(
        0.10 * total_training_timesteps
    )

    epsilon = epsilon_start

    # ---------------------------------------------------------
    # Training bookkeeping
    # ---------------------------------------------------------
    reward_history = deque(maxlen=10)

    training_updates = 0
    global_step = 0

    for episode in range(num_episodes):

        # Reset the environment
        observation, info = env.reset()

        # Create the initial 4-frame state
        state = state_manager.reset(observation)

        done = False
        total_reward = 0

        action_counts = {
            0: 0,  # NOOP
            1: 0,  # LEFT
            2: 0,  # RIGHT
        }

        episode_losses = []

        for step in range(max_steps_per_episode):

            # -------------------------------------------------
            # Calculate epsilon from global training timestep
            # -------------------------------------------------
            if global_step < epsilon_decay_steps:
                decay_progress = (
                    global_step / epsilon_decay_steps
                )

                epsilon = (
                    epsilon_start
                    + decay_progress
                    * (epsilon_end - epsilon_start)
                )
            else:
                epsilon = epsilon_end

            # -------------------------------------------------
            # Convert state to tensor
            # -------------------------------------------------
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0) / 255.0

            # -------------------------------------------------
            # Select action
            # -------------------------------------------------
            action = select_action(
                model,
                state_tensor,
                epsilon=epsilon
            )

            action_counts[action] += 1

            # -------------------------------------------------
            # Take action in Pong
            # -------------------------------------------------
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            # -------------------------------------------------
            # Create next state
            # -------------------------------------------------
            next_state = state_manager.step(observation)

            # -------------------------------------------------
            # Check episode termination
            # -------------------------------------------------
            done = terminated or truncated

            # -------------------------------------------------
            # Store experience
            # -------------------------------------------------
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # -------------------------------------------------
            # Train every 4 environment steps
            # -------------------------------------------------
            if (
                global_step >= training_start
                and len(replay_buffer) >= batch_size
                and global_step % train_frequency == 0
            ):

                loss = train_step(
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device
                )

                if loss is not None:

                    episode_losses.append(loss)

                    training_updates += 1

                    # Update target network periodically
                    if (
                        training_updates
                        % target_update_frequency
                        == 0
                    ):
                        update_target_network(
                            model,
                            target_model
                        )

            # -------------------------------------------------
            # Move to next state
            # -------------------------------------------------
            state = next_state

            total_reward += reward

            global_step += 1

            if done:
                break

        # -----------------------------------------------------
        # Episode statistics
        # -----------------------------------------------------
        average_loss = (
            sum(episode_losses) / len(episode_losses)
            if episode_losses
            else None
        )

        reward_history.append(total_reward)

        average_reward = (
            sum(reward_history)
            / len(reward_history)
        )

        print(
            f"Episode {episode + 1}: "
            f"Total reward = {total_reward}, "
            f"Average reward (last {len(reward_history)}) = "
            f"{average_reward:.2f}, "
            f"Replay buffer size = {len(replay_buffer)}, "
            f"Average Loss = {average_loss}, "
            f"Epsilon = {epsilon:.3f}, "
            f"Global steps = {global_step}, "
            f"Actions = {action_counts}"
        )

    # ---------------------------------------------------------
    # Save trained model
    # ---------------------------------------------------------
    torch.save(
        model.state_dict(),
        "pong_dqn_model.pth"
    )

    env.close()


if __name__ == "__main__":
    main()