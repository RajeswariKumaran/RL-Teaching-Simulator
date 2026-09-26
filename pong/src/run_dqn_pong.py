import random

import numpy as np
import torch

from src.action_selection import select_action
from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.replay_buffer import ReplayBuffer
from src.state import PongState
from src.target_network import update_target_network
from src.training import train_step


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_ACTIONS = 3

BATCH_SIZE = 32
TRAINING_START = 50_000
TRAIN_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 1_000

GAMMA = 0.99
LEARNING_RATE = 0.0001

TOTAL_TRAINING_TIMESTEPS = 500_000

REPLAY_BUFFER_CAPACITY = 100_000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 500_000

SEED = 42

CHECKPOINT_FREQUENCY = 100_000

MODEL_PATH = "pong_dqn_model.pth"

# ---------------------------------------------------------
# Resume configuration
# ---------------------------------------------------------

RESUME = False


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_epsilon(global_step):
    if global_step >= EPSILON_DECAY_STEPS:
        return EPSILON_END

    progress = global_step / EPSILON_DECAY_STEPS

    return EPSILON_START + progress * (
        EPSILON_END - EPSILON_START
    )


def save_checkpoint(
    model,
    optimizer,
    global_step,
    episode,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "episode": episode,
        },
        MODEL_PATH,
    )


def load_checkpoint(
    model,
    target_model,
    optimizer,
    device,
):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    target_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    global_step = checkpoint["global_step"]
    episode = checkpoint["episode"]

    return global_step, episode


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    set_seed(SEED)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    env = PongEnvironment(seed=SEED)
    state_manager = PongState()

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    model = DQN(
        num_actions=NUM_ACTIONS
    ).to(device)

    target_model = DQN(
        num_actions=NUM_ACTIONS
    ).to(device)

    target_model.eval()

    print(
        f"Model device: "
        f"{next(model.parameters()).device}"
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -----------------------------------------------------
    # Resume model if requested
    # -----------------------------------------------------

    if RESUME:
        global_step, episode = load_checkpoint(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"\nResuming training from checkpoint:"
        )
        print(
            f"Global steps: {global_step}"
        )
        print(
            f"Episode: {episode}"
        )
        print(
            "Replay buffer will be rebuilt "
            "from new experience."
        )

    else:
        update_target_network(
            model,
            target_model,
        )

        global_step = 0
        episode = 0

    # -----------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY,
    )

    # -----------------------------------------------------
    # Episode statistics
    # -----------------------------------------------------

    episode_reward = 0.0
    episode_steps = 0
    episode_losses = []

    episode_actions = {
        0: 0,
        1: 0,
        2: 0,
    }

    recent_rewards = []

    # -----------------------------------------------------
    # Environment reset
    # -----------------------------------------------------

    observation, _ = env.reset()
    state = state_manager.reset(observation)

    try:
        while global_step < TOTAL_TRAINING_TIMESTEPS:

            epsilon = get_epsilon(global_step)

            # -------------------------------------------------
            # Select action
            # -------------------------------------------------

            state_tensor = (
                torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(0)
                / 255.0
            )

            action = select_action(
                model,
                state_tensor,
                epsilon,
                num_actions=NUM_ACTIONS,
            )

            episode_actions[action] += 1

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

            done = terminated or truncated

            # -------------------------------------------------
            # Store experience
            # -------------------------------------------------

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            global_step += 1
            episode_steps += 1
            episode_reward += reward

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            if (
                global_step >= TRAINING_START
                and global_step % TRAIN_FREQUENCY == 0
            ):
                loss = train_step(
                    model=model,
                    replay_buffer=replay_buffer,
                    target_model=target_model,
                    optimizer=optimizer,
                    batch_size=BATCH_SIZE,
                    gamma=GAMMA,
                    device=device,
                )

                if loss is not None:
                    episode_losses.append(loss)

            # -------------------------------------------------
            # Target network
            # -------------------------------------------------

            if (
                global_step
                % TARGET_UPDATE_FREQUENCY
                == 0
            ):
                update_target_network(
                    model,
                    target_model,
                )

            # -------------------------------------------------
            # Episode completed
            # -------------------------------------------------

            if done:
                episode += 1

                recent_rewards.append(
                    episode_reward
                )

                if len(recent_rewards) > 10:
                    recent_rewards.pop(0)

                average_reward = (
                    sum(recent_rewards)
                    / len(recent_rewards)
                )

                average_loss = (
                    sum(episode_losses)
                    / len(episode_losses)
                    if episode_losses
                    else None
                )

                print(
                    f"Episode {episode}: "
                    f"Total reward = "
                    f"{episode_reward:.1f}, "
                    f"Average reward "
                    f"(last {len(recent_rewards)}) = "
                    f"{average_reward:.2f}, "
                    f"Replay buffer size = "
                    f"{len(replay_buffer)}, "
                    f"Average Loss = "
                    f"{average_loss}, "
                    f"Epsilon = "
                    f"{epsilon:.3f}, "
                    f"Global steps = "
                    f"{global_step}, "
                    f"Episode steps = "
                    f"{episode_steps}, "
                    f"Actions = "
                    f"{episode_actions}"
                )

                episode_reward = 0.0
                episode_steps = 0
                episode_losses = []

                episode_actions = {
                    0: 0,
                    1: 0,
                    2: 0,
                }

                observation, _ = env.reset()

                state = state_manager.reset(
                    observation
                )

            # -------------------------------------------------
            # Periodic checkpoint
            # -------------------------------------------------

            if (
                global_step
                % CHECKPOINT_FREQUENCY
                == 0
            ):
                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    episode,
                )

                print(
                    f"Checkpoint saved at "
                    f"{global_step} steps."
                )

    finally:
        env.close()

    # ---------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------

    save_checkpoint(
        model,
        optimizer,
        global_step,
        episode,
    )

    print("\nTraining complete.")
    print(
        f"Total training timesteps: "
        f"{global_step}"
    )
    print(
        f"Model saved to {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()