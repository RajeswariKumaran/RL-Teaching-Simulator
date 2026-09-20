from collections import deque

import numpy as np

from src.frame_stack import stack_frames


class PongState:
    """Maintain the four-frame state used by the Atari DQN."""

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, observation):
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(observation)
        return stack_frames(list(self.frames))

    def step(self, observation):
        self.frames.append(observation)
        return stack_frames(list(self.frames))
