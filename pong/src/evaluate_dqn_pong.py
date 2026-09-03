import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action


def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Using device:", device)

    # Create the trained model
    model = DQN().to(device)

    # Load the learned weights
    model.load_state_dict(
        torch.load(
            "pong_dqn_model.pth",
            map_location=device
        )
    )

    # Evaluation mode
    model.eval()

    env = PongEnvironment()
    state_manager = PongState()

    num_episodes = 1
    max_steps_per_episode = 1_000

    rewards = []

    for episode in range(num_episodes):

        observation, info = env.reset()
        state = state_manager.reset(observation)

        total_reward = 0
        action_counts = {}
        for step in range(max_steps_per_episode):

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0) / 255.0

            if step % 100 == 0:
                with torch.no_grad():
                    q_values = model(state_tensor)

                print(
                    "Q-values:",
                    q_values.squeeze(0).cpu().numpy()
                )
            # No exploration — choose the best action
            action = select_action(
                model,
                state_tensor,
                epsilon=0.0
            )
            action_counts[action] = action_counts.get(action, 0) + 1
            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            next_state = state_manager.step(observation)

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

        print(
            f"Evaluation Episode {episode + 1}: "
            f"Total reward = {total_reward}, "
            f"Actions = {action_counts}"
        )

    average_reward = sum(rewards) / len(rewards)

    print(
        f"\nAverage evaluation reward: {average_reward:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()