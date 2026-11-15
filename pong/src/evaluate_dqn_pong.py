import os
import torch
from collections import Counter

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 2000

CHECKPOINTS = {
    # "Vanilla 200K": "pong_dqn_model_200Ksteps.pth",
    # "Vanilla 500K (earlier)": "pong_dqn_model_500Ksteps.pth",
    # "Vanilla 500K (successful)": "pong_dqn_model_500K_Vanila.pth",
    # "Double DQN 500K": "pong_dqn_model_500K_DoubleDQN.pth",
    "Vanilla 1M": "pong_dqn_model.pth"
}


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model(model_path, device):

    model = DQN(num_actions=3).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    # Checkpoint created by the newer training script
    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    # Older models containing only state_dict
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ---------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------

def evaluate_model(model_path, device):

    model = load_model(
        model_path,
        device,
    )

    env = PongEnvironment()
    state_manager = PongState()

    episode_rewards = []
    episode_lengths = []

    total_action_counts = Counter()

    try:

        for episode in range(NUM_EPISODES):

            observation, _ = env.reset()

            state = state_manager.reset(
                observation
            )

            total_reward = 0.0

            for step in range(
                MAX_STEPS_PER_EPISODE
            ):

                # -----------------------------------------
                # Convert state to tensor
                # -----------------------------------------

                state_tensor = (
                    torch.as_tensor(
                        state,
                        dtype=torch.float32,
                        device=device,
                    )
                    .unsqueeze(0)
                    / 255.0
                )

                # -----------------------------------------
                # Greedy action
                # -----------------------------------------

                action = select_action(
                    model,
                    state_tensor,
                    epsilon=0.0,
                )

                total_action_counts[action] += 1

                # -----------------------------------------
                # Environment step
                # -----------------------------------------

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                total_reward += reward

                # -----------------------------------------
                # Update state
                # -----------------------------------------

                state = state_manager.step(
                    observation
                )

                if terminated or truncated:
                    break

            episode_rewards.append(
                total_reward
            )

            episode_lengths.append(
                step + 1
            )

    finally:
        env.close()

    average_reward = (
        sum(episode_rewards)
        / len(episode_rewards)
    )

    average_episode_length = (
        sum(episode_lengths)
        / len(episode_lengths)
    )

    return {
        "average_reward": average_reward,
        "average_episode_length":
            average_episode_length,
        "episode_rewards": episode_rewards,
        "action_counts":
            dict(total_action_counts),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(
        f"Evaluating {len(CHECKPOINTS)} models"
    )
    print(
        f"Episodes per model: {NUM_EPISODES}"
    )
    print(
        f"Maximum steps per episode: "
        f"{MAX_STEPS_PER_EPISODE}"
    )
    print()

    results = {}

    # -----------------------------------------------------
    # Evaluate every checkpoint
    # -----------------------------------------------------

    for name, model_path in CHECKPOINTS.items():

        print("=" * 70)
        print(f"Evaluating: {name}")
        print(f"File: {model_path}")
        print("=" * 70)

        if not os.path.exists(model_path):

            print(
                f"WARNING: File not found: "
                f"{model_path}"
            )
            print()

            continue

        result = evaluate_model(
            model_path,
            device,
        )

        results[name] = result

        print(
            f"Average reward: "
            f"{result['average_reward']:.2f}"
        )

        print(
            f"Average episode length: "
            f"{result['average_episode_length']:.1f}"
        )

        print(
            f"Action counts: "
            f"{result['action_counts']}"
        )

        print(
            f"Episode rewards: "
            f"{result['episode_rewards']}"
        )

        print()

    # -----------------------------------------------------
    # Comparison table
    # -----------------------------------------------------

    print("\n")
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    print(
        f"{'Model':<30}"
        f"{'Avg Reward':>12}"
        f"{'Avg Steps':>12}"
        f"{'NOOP':>10}"
        f"{'LEFT':>10}"
        f"{'RIGHT':>10}"
    )

    print("-" * 100)

    for name, result in results.items():

        actions = result["action_counts"]

        print(
            f"{name:<30}"
            f"{result['average_reward']:>12.2f}"
            f"{result['average_episode_length']:>12.1f}"
            f"{actions.get(0, 0):>10}"
            f"{actions.get(1, 0):>10}"
            f"{actions.get(2, 0):>10}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()