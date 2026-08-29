import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action
from src.replay_buffer import ReplayBuffer


def main():

    env = PongEnvironment()
    state_manager = PongState()
    model = DQN()
    replay_buffer = ReplayBuffer(capacity=10_000)

    observation, info = env.reset()
    state = state_manager.reset(observation)

    done = False
    total_reward = 0
    step = 0
    epsilon = 0.5

    while not done and step < 100:

        # Convert the current state to a PyTorch tensor
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        # Select an action using epsilon-greedy exploration
        action = select_action(
            model,
            state_tensor,
            epsilon=epsilon
        )

        # Take the action in Pong
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        # Create the next RL state
        next_state = state_manager.step(observation)

        # Check whether the episode has ended
        done = terminated or truncated

        # Store the experience
        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Move to the next state
        state = next_state

        total_reward += reward
        step += 1

        print(
            f"Step: {step}, "
            f"Action: {action}, "
            f"Reward: {reward}, "
            f"Total reward: {total_reward}"
        )

    print("\nEpisode finished")
    print("Total reward:", total_reward)
    print("Experiences stored:", len(replay_buffer))

    env.close()


if __name__ == "__main__":
    main()