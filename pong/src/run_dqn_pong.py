import torch

from src.dqn import DQN
from src.pong_environment import PongEnvironment
from src.state import PongState
from src.action_selection import select_action


def main():

    env = PongEnvironment()
    state_manager = PongState()
    model = DQN()

    observation, info = env.reset()
    state = state_manager.reset(observation)

    done = False
    total_reward = 0
    step = 0

    while not done and step < 100:

        # Convert the current state to a PyTorch tensor
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0)

        # Let the DQN estimate Q-values
        epsilon = 0.5

        action = select_action(
            model,
            state_tensor,
            epsilon=epsilon
        )

        # Take the action in Pong
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        # Convert the new observation into the next RL state
        state = state_manager.step(observation)

        total_reward += reward
        step += 1

        done = terminated or truncated

        print(
            f"Step: {step}, "
            f"Action: {action}, "
            f"Reward: {reward}, "
            f"Total reward: {total_reward}"
        )

    print("\nEpisode finished")
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()