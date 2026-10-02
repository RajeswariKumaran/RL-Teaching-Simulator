import torch
import numpy as np
from collections import deque

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "pong_dqn_model.pth"

MAX_STEPS = 2000
NUM_EPISODES = 5

HISTORY_LENGTH = 20

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ============================================================
# Helpers
# ============================================================

ACTION_NAMES = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
}


def get_q_values(model, state):
    """
    Run the current state through the network and return
    Q-values for NOOP, LEFT and RIGHT.
    """

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0) / 255.0

    with torch.no_grad():
        q_values = model(state_tensor)[0]

    return q_values.cpu().numpy()


def calculate_margin(q_values):
    """
    Difference between the largest and second-largest Q-value.
    """

    sorted_q = np.sort(q_values)[::-1]

    return float(sorted_q[0] - sorted_q[1])


def count_action_changes(history):
    """
    Count how many times the selected action changed
    within the supplied history.
    """

    if len(history) < 2:
        return 0

    actions = [item["action"] for item in history]

    return sum(
        actions[i] != actions[i - 1]
        for i in range(1, len(actions))
    )


def get_history_metric(history, steps_back):
    """
    Get a metric from a specific number of steps before
    the current event.

    steps_back=0  -> event itself
    steps_back=5  -> five steps before event
    steps_back=10 -> ten steps before event
    """

    index = len(history) - 1 - steps_back

    if index < 0:
        return None

    return history[index]


def print_event(event_name, step, history):

    print()
    print("=" * 80)
    print(f"{event_name} at step {step}")
    print("=" * 80)

    action_changes = count_action_changes(history)

    margins = [item["margin"] for item in history]

    print(
        f"Action changes in previous "
        f"{len(history)} steps: {action_changes}"
    )

    print(
        f"Maximum margin in history: "
        f"{max(margins):.3f}"
    )

    event_item = get_history_metric(history, 0)
    five_back = get_history_metric(history, 5)
    ten_back = get_history_metric(history, 10)

    if event_item is not None:
        print(
            f"Margin at event:             "
            f"{event_item['margin']:.3f}"
        )

    if five_back is not None:
        print(
            f"Margin 5 steps before:        "
            f"{five_back['margin']:.3f}"
        )

    if ten_back is not None:
        print(
            f"Margin 10 steps before:       "
            f"{ten_back['margin']:.3f}"
        )

    print()

    print(
        f"{'Step':>6} "
        f"{'Q[N]':>9} "
        f"{'Q[L]':>9} "
        f"{'Q[R]':>9} "
        f"{'Margin':>9} "
        f"{'Action':>8} "
        f"{'Reward':>8}"
    )

    print("-" * 80)

    for item in history:

        q = item["q_values"]

        print(
            f"{item['step']:6d} "
            f"{q[0]:9.3f} "
            f"{q[1]:9.3f} "
            f"{q[2]:9.3f} "
            f"{item['margin']:9.3f} "
            f"{ACTION_NAMES[item['action']]:>8} "
            f"{item['reward']:8.1f}"
        )


def record_event_statistics(
    history,
    margin_list,
    max_margin_list,
    margin_5_list,
    margin_10_list,
    action_change_list
):
    """
    Store diagnostic statistics for a reward event.
    """

    event_item = get_history_metric(history, 0)
    five_back = get_history_metric(history, 5)
    ten_back = get_history_metric(history, 10)

    if event_item is None:
        return

    margin_list.append(event_item["margin"])

    max_margin_list.append(
        max(item["margin"] for item in history)
    )

    if five_back is not None:
        margin_5_list.append(five_back["margin"])

    if ten_back is not None:
        margin_10_list.append(ten_back["margin"])

    action_change_list.append(
        count_action_changes(history)
    )


def print_summary(
    name,
    margins,
    max_margins,
    margin_5,
    margin_10,
    action_changes
):

    print()
    print(name)

    if not margins:
        print("  No events found.")
        return

    print(f"  Events:                  {len(margins)}")

    print(
        f"  Average event margin:    "
        f"{np.mean(margins):.4f}"
    )

    print(
        f"  Median event margin:     "
        f"{np.median(margins):.4f}"
    )

    print(
        f"  Average maximum margin:  "
        f"{np.mean(max_margins):.4f}"
    )

    if margin_5:
        print(
            f"  Average margin -5:       "
            f"{np.mean(margin_5):.4f}"
        )

    if margin_10:
        print(
            f"  Average margin -10:      "
            f"{np.mean(margin_10):.4f}"
        )

    print(
        f"  Average action changes:   "
        f"{np.mean(action_changes):.2f}"
    )


# ============================================================
# Load model
# ============================================================

print(f"Using device: {DEVICE}")

model = DQN(num_actions=3).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f"Loaded model: {MODEL_PATH}")


# ============================================================
# Environment
# ============================================================

env = PongEnvironment(render_mode=None)
state_manager = PongState()


# ============================================================
# Statistics
# ============================================================

# Miss statistics
miss_margins = []
miss_max_margins = []
miss_margin_5 = []
miss_margin_10 = []
miss_action_changes = []

# Success statistics
success_margins = []
success_max_margins = []
success_margin_5 = []
success_margin_10 = []
success_action_changes = []


# ============================================================
# Run diagnostic episodes
# ============================================================

for episode in range(1, NUM_EPISODES + 1):

    observation, info = env.reset()

    state = state_manager.reset(observation)

    history = deque(maxlen=HISTORY_LENGTH)

    episode_reward = 0

    print()
    print(
        f"\n{'=' * 80}\n"
        f"EPISODE {episode}\n"
        f"{'=' * 80}"
    )

    for step in range(1, MAX_STEPS + 1):

        # ----------------------------------------------------
        # Q-values and action
        # ----------------------------------------------------

        q_values = get_q_values(model, state)

        action = int(np.argmax(q_values))

        margin = calculate_margin(q_values)

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        next_state = state_manager.step(next_observation)

        episode_reward += reward

        # ----------------------------------------------------
        # Store current decision
        # ----------------------------------------------------

        event = {
            "step": step,
            "q_values": q_values,
            "margin": margin,
            "action": action,
            "reward": reward,
        }

        history.append(event)

        # ----------------------------------------------------
        # Diagnose reward events
        # ----------------------------------------------------

        if reward == -1:

            record_event_statistics(
                history,
                miss_margins,
                miss_max_margins,
                miss_margin_5,
                miss_margin_10,
                miss_action_changes,
            )

            print_event(
                "MISS (-1)",
                step,
                list(history)
            )

        elif reward == 1:

            record_event_statistics(
                history,
                success_margins,
                success_max_margins,
                success_margin_5,
                success_margin_10,
                success_action_changes,
            )

            print_event(
                "SUCCESS (+1)",
                step,
                list(history)
            )

        # ----------------------------------------------------
        # Move to next state
        # ----------------------------------------------------

        state = next_state

        if terminated or truncated:
            break

    print(
        f"\nEpisode {episode} finished: "
        f"reward={episode_reward}"
    )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print_summary(
    "MISSES (-1)",
    miss_margins,
    miss_max_margins,
    miss_margin_5,
    miss_margin_10,
    miss_action_changes,
)

print_summary(
    "SUCCESSES (+1)",
    success_margins,
    success_max_margins,
    success_margin_5,
    success_margin_10,
    success_action_changes,
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)