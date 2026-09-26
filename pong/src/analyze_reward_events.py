import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState


MODEL_PATH = "pong_dqn_model.pth"

NUM_ACTIONS = 3
NUM_EPISODES = 3
MAX_STEPS = 2000

ACTION_NAMES = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
}


def load_model(device):
    model = DQN(num_actions=NUM_ACTIONS).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Episodes:", NUM_EPISODES)
    print()

    model = load_model(device)

    env = PongEnvironment()
    state_manager = PongState()

    for episode in range(1, NUM_EPISODES + 1):

        observation, _ = env.reset()
        state = state_manager.reset(observation)

        total_reward = 0.0
        reward_events = []

        print("=" * 75)
        print(f"EPISODE {episode}")
        print("=" * 75)

        for step in range(1, MAX_STEPS + 1):

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
                action = int(
                    torch.argmax(q_values, dim=1).item()
                )

            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )

            total_reward += reward

            # Only print when the environment produces a non-zero reward.
            if reward != 0:

                q = q_values[0].detach().cpu().numpy()

                print(
                    f"Step {step:4d} | "
                    f"Reward {reward:+.0f} | "
                    f"Action {ACTION_NAMES[action]:>5} | "
                    f"Q[NOOP]={q[0]:8.3f} | "
                    f"Q[LEFT]={q[1]:8.3f} | "
                    f"Q[RIGHT]={q[2]:8.3f}"
                )

                reward_events.append(
                    {
                        "step": step,
                        "reward": reward,
                        "action": action,
                    }
                )

            state = state_manager.step(next_observation)

            if terminated or truncated:
                break

        print()
        print(
            f"Episode {episode} finished: "
            f"reward={total_reward:.1f}, "
            f"steps={step}"
        )

        print(
            f"Non-zero reward events: {len(reward_events)}"
        )

        print()

    env.close()

    print("=" * 75)
    print("Reward-event analysis complete")
    print("=" * 75)


if __name__ == "__main__":
    main()