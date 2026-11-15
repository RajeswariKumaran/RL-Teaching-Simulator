import cv2
import numpy as np
import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState


MODEL_PATH = "pong_dqn_model.pth"
NUM_ACTIONS = 3
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


def find_objects(frame):
    """
    Find the ball and the player's paddle in an 84x84 grayscale frame.

    Returns:
        ball_x, ball_y, paddle_x, paddle_y
    """

    # Threshold bright pixels.
    binary = (frame > 180).astype(np.uint8)

    # Ignore the top scoreboard area.
    binary[:10, :] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    ball = None
    paddle = None

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Ignore very small noise and large court/border structures.
        if area < 2 or area > 100:
            continue

        # Ball: small approximately square component.
        if (
            w <= 5
            and h <= 5
            and area <= 20
        ):
            cx, cy = centroids[i]

            # Prefer objects in the actual playing area.
            if 10 < cy < 80:
                ball = (cx, cy)

        # Paddle: narrow and vertically elongated.
        if (
            h >= 5
            and h <= 15
            and w <= 5
            and area >= 8
        ):
            cx, cy = centroids[i]

            # Our paddle is on the right half.
            if cx > 55:
                paddle = (cx, cy)

    return ball, paddle


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading model:", MODEL_PATH)

    model = load_model(device)

    env = PongEnvironment(render_mode="rgb_array")
    state_manager = PongState()

    observation, info = env.reset()
    state = state_manager.reset(observation)

    total_reward = 0.0

    print()
    print(
        f"{'Step':>5} "
        f"{'Ball':>14} "
        f"{'Paddle':>14} "
        f"{'DistY':>7} "
        f"{'Action':>7} "
        f"{'Reward':>7}"
    )
    print("-" * 65)

    for step in range(1, MAX_STEPS + 1):

        # Last frame is the most recent observation.
        frame = state[-1]

        ball, paddle = find_objects(frame)

        # Convert state to tensor.
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        state = state_manager.step(next_observation)

        # Print every 10 decisions.
        if step % 10 == 0:

            if ball is not None:
                ball_text = f"({ball[0]:5.1f},{ball[1]:5.1f})"
            else:
                ball_text = "   N/A"

            if paddle is not None:
                paddle_text = f"({paddle[0]:5.1f},{paddle[1]:5.1f})"
            else:
                paddle_text = "   N/A"

            if ball is not None and paddle is not None:
                dist_y = ball[1] - paddle[1]
                dist_text = f"{dist_y:7.1f}"
            else:
                dist_text = "    N/A"

            print(
                f"{step:5d} "
                f"{ball_text:>14} "
                f"{paddle_text:>14} "
                f"{dist_text:>7} "
                f"{ACTION_NAMES[action]:>7} "
                f"{reward:7.1f}"
            )

        if terminated or truncated:
            print()
            print("Episode ended at step:", step)
            break

    env.close()

    print()
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()