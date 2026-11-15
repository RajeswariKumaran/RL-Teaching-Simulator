from src.pong_environment import PongEnvironment
from src.state import PongState


def main():

    env = PongEnvironment()

    state_manager = PongState()

    observation, info = env.reset()

    state = state_manager.reset(observation)

    print("Initial state shape:", state.shape)

    for step in range(10):

        action = 2  # RIGHT

        observation, reward, terminated, truncated, info = env.step(action)

        state = state_manager.step(observation)

        print(
            f"Step {step + 1}: "
            f"state={state.shape}, "
            f"reward={reward}, "
            f"terminated={terminated}"
        )

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()