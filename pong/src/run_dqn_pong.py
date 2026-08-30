import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action
from src.replay_buffer import ReplayBuffer
# from src.train_step import train_step
from src.training import train_step
from collections import deque


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
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )
    replay_buffer = ReplayBuffer(capacity=10_000)

    batch_size = 32
    # training_start = 1_000
    training_start = 300
    train_frequency = 4
    gamma = 0.99

    num_episodes = 100
    # num_episodes = 5
    max_steps_per_episode = 1_000
    # max_steps_per_episode = 300

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    target_update_frequency = 1_000

    reward_history = deque(maxlen=10)
    training_updates = 0
    for episode in range(num_episodes):

        # Reset the environment at the start of each episode
        observation, info = env.reset()
        state = state_manager.reset(observation)

        done = False
        total_reward = 0

        episode_losses = []
        for step in range(max_steps_per_episode):

            # Convert the current state to a PyTorch tensor
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # Select an action using epsilon-greedy exploration
            action = select_action(
                model,
                state_tensor,
                epsilon=epsilon
            )

            # Take the action in Pong
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Create the next RL state
            next_state = state_manager.step(observation)

            # Check whether the episode has ended
            done = terminated or truncated

            # Store the experience
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Start training once enough experiences have been collected
            # if len(replay_buffer) >= training_start:
            if (
                len(replay_buffer) >= training_start
                and step % train_frequency == 0
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
                    if training_updates % target_update_frequency == 0:
                        target_model.load_state_dict(model.state_dict())

            # Move to the next state
            state = next_state

            total_reward += reward

            if done:
                break

        average_loss = (
            sum(episode_losses) / len(episode_losses)
            if episode_losses
            else None
        )
        # print("Average Loss:", average_loss)
        reward_history.append(total_reward)
        average_reward = sum(reward_history) / len(reward_history)
        print(
            f"Episode {episode + 1}: "
            f"Total reward = {total_reward}, "
            f"Average reward (last {len(reward_history)}) = "
            f"{average_reward:.2f}, "
            f"Replay buffer size = {len(replay_buffer)},"
            f"Average Loss = {average_loss},"
            f"Epsilon = {epsilon:.3f}"
        )
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

    env.close()


if __name__ == "__main__":
    main()