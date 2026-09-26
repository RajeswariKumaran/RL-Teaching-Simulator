import torch
from collections import Counter

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VANILLA_MODEL_PATH = "pong_dqn_model_500K_Vanila.pth"
DOUBLE_DQN_MODEL_PATH = "pong_dqn_model_500K_DoubleDQN.pth"

NUM_EPISODES = 5
MAX_STEPS_PER_EPISODE = 2000

NUM_ACTIONS = 3

ACTION_NAMES = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
}


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

def load_model(path, device):

    model = DQN(num_actions=NUM_ACTIONS).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# ---------------------------------------------------------
# Get Q-values
# ---------------------------------------------------------

def get_q_values(model, state, device):

    state_tensor = (
        torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        / 255.0
    )

    with torch.no_grad():
        q_values = model(state_tensor)

    q_values = q_values.squeeze(0)

    action = q_values.argmax().item()

    return q_values.cpu().tolist(), action


# ---------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------

def diagnose():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------
    # Load both models
    # -----------------------------------------------------

    vanilla_model = load_model(
        VANILLA_MODEL_PATH,
        device,
    )

    double_model = load_model(
        DOUBLE_DQN_MODEL_PATH,
        device,
    )

    print("Loaded both models.")

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    env = PongEnvironment()
    state_manager = PongState()

    total_states = 0
    different_actions = 0

    vanilla_action_counts = Counter()
    double_action_counts = Counter()

    vanilla_rewards = []
    double_rewards = []

    try:

        for episode in range(NUM_EPISODES):

            print()
            print("=" * 70)
            print(f"EPISODE {episode + 1}")
            print("=" * 70)

            observation, _ = env.reset()

            state = state_manager.reset(
                observation
            )

            vanilla_episode_reward = 0.0
            double_episode_reward = 0.0

            # The environment is actually being controlled by
            # the vanilla model. We compare what Double DQN
            # WOULD have done on the exact same states.
            for step in range(MAX_STEPS_PER_EPISODE):

                # -------------------------------------------------
                # Get both models' Q-values
                # -------------------------------------------------

                vanilla_q, vanilla_action = get_q_values(
                    vanilla_model,
                    state,
                    device,
                )

                double_q, double_action = get_q_values(
                    double_model,
                    state,
                    device,
                )

                vanilla_action_counts[
                    vanilla_action
                ] += 1

                double_action_counts[
                    double_action
                ] += 1

                total_states += 1

                if vanilla_action != double_action:
                    different_actions += 1

                # -------------------------------------------------
                # Take VANILLA action
                # -------------------------------------------------

                observation, reward, terminated, truncated, _ = (
                    env.step(vanilla_action)
                )

                vanilla_episode_reward += reward
                double_episode_reward += reward

                # -------------------------------------------------
                # Print reward events
                # -------------------------------------------------

                if reward != 0:

                    print()
                    print(
                        f"Step {step + 1}: "
                        f"REWARD = {reward:+.0f}"
                    )

                    print(
                        "Vanilla DQN:"
                    )

                    print(
                        f"  NOOP  = {vanilla_q[0]: .4f}"
                    )

                    print(
                        f"  LEFT  = {vanilla_q[1]: .4f}"
                    )

                    print(
                        f"  RIGHT = {vanilla_q[2]: .4f}"
                    )

                    print(
                        f"  Action = "
                        f"{ACTION_NAMES[vanilla_action]}"
                    )

                    print(
                        "Double DQN:"
                    )

                    print(
                        f"  NOOP  = {double_q[0]: .4f}"
                    )

                    print(
                        f"  LEFT  = {double_q[1]: .4f}"
                    )

                    print(
                        f"  RIGHT = {double_q[2]: .4f}"
                    )

                    print(
                        f"  Action = "
                        f"{ACTION_NAMES[double_action]}"
                    )

                state = state_manager.step(
                    observation
                )

                if terminated or truncated:
                    break

            vanilla_rewards.append(
                vanilla_episode_reward
            )

            double_rewards.append(
                double_episode_reward
            )

            print()
            print(
                f"Episode {episode + 1} complete"
            )

            print(
                f"Steps: {step + 1}"
            )

            print(
                f"Reward received: "
                f"{vanilla_episode_reward:.1f}"
            )

            print(
                f"Vanilla actions: "
                f"{dict(vanilla_action_counts)}"
            )

            print(
                f"Double actions: "
                f"{dict(double_action_counts)}"
            )

    finally:
        env.close()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    agreement = (
        100.0
        * (total_states - different_actions)
        / total_states
    )

    disagreement = (
        100.0
        * different_actions
        / total_states
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"States compared: {total_states}"
    )

    print(
        f"Same action: "
        f"{total_states - different_actions} "
        f"({agreement:.2f}%)"
    )

    print(
        f"Different action: "
        f"{different_actions} "
        f"({disagreement:.2f}%)"
    )

    print()

    print(
        "Vanilla action counts:"
    )
    print(
        dict(vanilla_action_counts)
    )

    print()

    print(
        "Double DQN action counts:"
    )
    print(
        dict(double_action_counts)
    )

    print()

    print(
        "Vanilla rewards:"
    )
    print(vanilla_rewards)

    print()

    print(
        "Rewards experienced while "
        "following vanilla policy:"
    )
    print(double_rewards)


if __name__ == "__main__":
    diagnose()