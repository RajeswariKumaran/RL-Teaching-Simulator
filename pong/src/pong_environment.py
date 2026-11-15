import gymnasium as gym
import ale_py

from gymnasium.wrappers import AtariPreprocessing

gym.register_envs(ale_py)


class PongEnvironment:
    """Small Atari Pong adapter used by the DQN."""

    ACTION_NOOP = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2

    def __init__(self, render_mode=None, seed=None):
        env = gym.make(
            "ALE/Pong-v5",
            render_mode=render_mode,
            frameskip=1,
        )

        # AtariPreprocessing performs the important Atari preprocessing:
        # frame skipping + max pooling + grayscale + 84x84 resize.
        self.env = AtariPreprocessing(
            env,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        self.seed = seed
        self._needs_fire = True

        # Once Pong has started, FIRE is not a useful control action.
        self.actions = {
            self.ACTION_NOOP: 0,
            self.ACTION_LEFT: 3,
            self.ACTION_RIGHT: 2,
        }

    def reset(self):
        observation, info = self.env.reset(seed=self.seed)
        self.seed = None

        # Pong needs FIRE after a real environment reset.
        if self._needs_fire:
            observation, _, terminated, truncated, info = self.env.step(1)
            if terminated or truncated:
                observation, info = self.env.reset()
            self._needs_fire = False

        return observation, info

    def step(self, action):
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        observation, reward, terminated, truncated, info = self.env.step(
            self.actions[action]
        )

        if terminated or truncated:
            self._needs_fire = True

        return observation, reward, terminated, truncated, info

    def close(self):
        self.env.close()
