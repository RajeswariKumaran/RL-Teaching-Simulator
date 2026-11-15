import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = "pong_dqn_model.pth"

NUM_ACTIONS = 3
MAX_STEPS = 2000

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = DQN(num_actions=NUM_ACTIONS).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f"Using device: {DEVICE}")
print(f"Loaded model: {MODEL_PATH}")


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

env = PongEnvironment(
    seed=42,
    render_mode="human",
)

state_manager = PongState()


# ---------------------------------------------------------
# Run one episode
# ---------------------------------------------------------

observation, _ = env.reset()

state = state_manager.reset(observation)

total_reward = 0.0

try:

    for step in range(1, MAX_STEPS + 1):

        # -------------------------------------------------
        # Convert state to tensor
        # -------------------------------------------------

        state_tensor = (
            torch.as_tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            )
            .unsqueeze(0)
            / 255.0
        )

        # -------------------------------------------------
        # Greedy action
        # epsilon = 0 means no random exploration
        # -------------------------------------------------

        # with torch.no_grad():
        #     action = select_action(
        #         model,
        #         state_tensor,
        #         epsilon=0.0,
        #         num_actions=NUM_ACTIONS,
        #     )

        with torch.no_grad():
            q_values = model(state_tensor)

            action = q_values.argmax(dim=1).item()

            q_noop = q_values[0, 0].item()
            q_left = q_values[0, 1].item()
            q_right = q_values[0, 2].item()

        # -------------------------------------------------
        # Environment step
        # -------------------------------------------------

        (
            next_observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        next_state = state_manager.step(
            next_observation
        )

        total_reward += reward

        # print(
        #     f"Step {step:4d} | "
        #     f"Action {action} | "
        #     f"Reward {reward:+.1f} | "
        #     f"Total {total_reward:+.1f}"
        # )

        q_sorted = sorted(
            [q_noop, q_left, q_right],
            reverse=True,
        )

        q_margin = q_sorted[0] - q_sorted[1]
        print(
            f"Step {step:4d} | "
            f"Q[NOOP]={q_noop:+.3f} | "
            f"Q[LEFT]={q_left:+.3f} | "
            f"Q[RIGHT]={q_right:+.3f} | "
            f"Margin={q_margin:.3f} | "
            f"Action={action} | "
            f"Reward={reward:+.1f} | "
            f"Total={total_reward:+.1f}"
        )
        state = next_state

        if terminated or truncated:
            print("\nEpisode finished.")
            break

finally:
    env.close()


print("\n======================================")
print("Visualization complete")
print(f"Total reward: {total_reward:.1f}")
print(f"Steps: {step}")
print("======================================")