import torch
from collections import Counter

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action


MODEL_PATH = "pong_dqn_model.pth"
NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 1000


def evaluate():
    # Select device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Create environment and state manager
    env = PongEnvironment()
    state_manager = PongState()

    # Create model
    model = DQN(num_actions=3).to(device)

    # Load trained model
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    episode_rewards = []

    try:
        for episode in range(NUM_EPISODES):
            observation, _ = env.reset()
            state = state_manager.reset(observation)

            total_reward = 0.0
            action_counts = Counter()

            for step in range(MAX_STEPS_PER_EPISODE):

                # Convert state to tensor
                state_tensor = (
                    torch.tensor(
                        state,
                        dtype=torch.float32,
                        device=device,
                    )
                    .unsqueeze(0)
                    / 255.0
                )

                # Greedy action selection
                with torch.no_grad():
                    q_values = model(state_tensor)

                action = select_action(
                    q_values,
                    epsilon=0.0,
                )

                action_counts[action] += 1

                # Take action
                observation, reward, terminated, truncated, _ = env.step(
                    action
                )

                total_reward += reward

                # Update state
                state = state_manager.step(observation)

                if terminated or truncated:
                    break

            episode_rewards.append(total_reward)

            print(
                f"Episode {episode + 1}: "
                f"Total reward = {total_reward}, "
                f"Steps = {step + 1}, "
                f"Actions = {dict(action_counts)}"
            )

    finally:
        env.close()

    average_reward = sum(episode_rewards) / len(episode_rewards)

    print("\nEvaluation complete")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Episode rewards: {episode_rewards}")


if __name__ == "__main__":
    evaluate()