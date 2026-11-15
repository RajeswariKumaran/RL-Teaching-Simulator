import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def main():

    env = gym.make(
        "ALE/Pong-v5",
        render_mode="human"
    )

    observation, info = env.reset()

    done = False

    while not done:

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()