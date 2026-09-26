import matplotlib.pyplot as plt
import torch

from src.pong_environment import PongEnvironment
from src.state import PongState


def main():
    env = PongEnvironment(render_mode="rgb_array")
    state_manager = PongState()

    observation, info = env.reset()
    state = state_manager.reset(observation)

    # Advance the game a little so the four frames are not just
    # the duplicated initial frame.
    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(2)  # RIGHT
        state = state_manager.step(observation)

        if terminated or truncated:
            break

    print("State type:", type(state))
    print("State shape:", state.shape)
    print("State dtype:", state.dtype)

    # Display the 4 frames that the DQN receives.
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for i in range(4):
        axes[i].imshow(state[i], cmap="gray")
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("pong_state_4_frames.png", dpi=150)
    plt.show()

    env.close()


if __name__ == "__main__":
    main()