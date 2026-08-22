import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class PongEnvironment:

    def __init__(self, render_mode=None):

        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode=render_mode
        )

        # Our agent only learns these three actions.
        self.actions = {
            0: 0,  # NOOP
            1: 3,  # LEFT
            2: 2,  # RIGHT
        }

    def reset(self):

        observation, info = self.env.reset()

        # Automatically serve the ball.
        observation, reward, terminated, truncated, info = (
            self.env.step(1)  # FIRE
        )

        return observation, info

    def step(self, action):

        if action not in self.actions:
            raise ValueError(
                f"Invalid action: {action}"
            )

        # Translate our action into the Atari action.
        atari_action = self.actions[action]

        return self.env.step(atari_action)

    def close(self):

        self.env.close()