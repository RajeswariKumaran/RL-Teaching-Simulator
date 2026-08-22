import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def main():

    env = gym.make("ALE/Pong-v5")

    observation, info = env.reset()

    print("Observation type:")
    print(type(observation))

    print("\nObservation shape:")
    print(observation.shape)

    print("\nAction space:")
    print(env.action_space)

    print("\nObservation space:")
    print(env.observation_space)

    print("\nNumber of actions:")
    print(env.action_space.n)

    print("\nActions:")
    for action in range(env.action_space.n):
        print(action)

    print("\nAction meanings:")
    for action in range(env.action_space.n):
        print(
            action,
            env.unwrapped.get_action_meanings()[action]
        )
    env.close()


if __name__ == "__main__":
    main()