import gymnasium as gym
import ale_py

import numpy as np


gym.register_envs(ale_py)


class PongEnvironment:

    def __init__(self, render_mode=None, skip=4):

        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode=render_mode,
            frameskip=1,
        )

        self.skip = skip

        self.actions = {
            0: 0,  # NOOP
            1: 3,  # LEFT
            2: 2,  # RIGHT
        }

    def reset(self):

        observation, info = self.env.reset()

        # Start the Atari game by firing.
        observation, reward, terminated, truncated, info = (
            self.env.step(1)
        )

        return observation, info

    def step(self, action):

        if action not in self.actions:
            raise ValueError(
                f"Invalid action: {action}"
            )

        atari_action = self.actions[action]

        total_reward = 0.0

        observations = []

        terminated = False
        truncated = False
        info = {}

        for frame in range(self.skip):

            observation, reward, terminated, truncated, info = (
                self.env.step(atari_action)
            )

            total_reward += reward

            # Keep observations from the last two frames.
            if frame >= self.skip - 2:
                observations.append(observation)

            if terminated or truncated:
                break

        # Default: use the last observation returned by Atari.
        max_observation = observation

        # Standard max-pooling when two observations
        # from the final frames are available.
        if len(observations) >= 2:
            max_observation = np.maximum(
                observations[-2],
                observations[-1]
            )

        return (
            max_observation,
            total_reward,
            terminated,
            truncated,
            info,
        )
    def close(self):
        self.env.close()