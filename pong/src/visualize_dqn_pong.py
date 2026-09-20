import imageio.v2 as imageio
import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState


MODEL_PATH = "pong_dqn_model.pth"

NUM_ACTIONS = 3
MAX_STEPS = 2000

# Video playback speed.
FPS = 30

VIDEO_PATH = "pong_dqn_500k_episode.mp4"

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

    print(f"Using device: {device}")

    model = load_model(device)

    # Use RGB-array rendering so we can save the actual Atari frames.
    env = PongEnvironment(render_mode="rgb_array")

    state_manager = PongState()

    observation, info = env.reset()
    state = state_manager.reset(observation)

    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}

    print()
    print("=" * 60)
    print("Recording GREEDY evaluation")
    print("Epsilon = 0.0")
    print(f"Maximum steps = {MAX_STEPS}")
    print(f"Output video = {VIDEO_PATH}")
    print("=" * 60)
    print()

    # Write frames directly to the video instead of keeping
    # all 2,000 frames in memory.
    with imageio.get_writer(
        VIDEO_PATH,
        fps=FPS,
        codec="libx264",
    ) as video:

        for step in range(1, MAX_STEPS + 1):

            # Save the current Atari frame.
            frame = env.env.render()

            if frame is not None:
                video.append_data(frame)

            # Convert NumPy state to PyTorch tensor.
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # Greedy action.
            with torch.no_grad():
                q_values = model(state_tensor)
                action = int(
                    torch.argmax(q_values, dim=1).item()
                )

            action_counts[action] += 1

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            state = state_manager.step(next_observation)

            if step % 100 == 0:
                q = q_values[0].detach().cpu().numpy()

                print(
                    f"Step {step:4d} | "
                    f"Action: {action} ({ACTION_NAMES[action]:5s}) | "
                    f"Reward: {total_reward: .1f} | "
                    f"Q: "
                    f"NOOP={q[0]: .2f}, "
                    f"LEFT={q[1]: .2f}, "
                    f"RIGHT={q[2]: .2f}"
                )

            if terminated or truncated:
                print(
                    f"\nEpisode ended naturally at step {step}."
                )
                break

    env.close()

    print()
    print("=" * 60)
    print("Recording complete")
    print("=" * 60)
    print(f"Total reward: {total_reward}")
    print(f"Steps: {step}")
    print(f"Actions: {action_counts}")
    print(f"Video saved to: {VIDEO_PATH}")
    print()


if __name__ == "__main__":
    main()