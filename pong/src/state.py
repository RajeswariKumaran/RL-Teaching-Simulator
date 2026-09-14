from collections import deque

from src.frame_stack import stack_frames
from src.preprocess import preprocess_frame


from collections import deque

import numpy as np

from src.frame_stack import stack_frames
from src.preprocess import preprocess_frame


class PongState:

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        self.previous_observation = None

    def reset(self, observation):

        processed = preprocess_frame(observation)

        self.frames.clear()

        for _ in range(self.num_frames):
            self.frames.append(processed)

        self.previous_observation = observation

        return stack_frames(list(self.frames))

    def step(self, observation):

        # Max-pool the current observation with the
        # previous observation, pixel by pixel.
        max_observation = np.maximum(
            self.previous_observation,
            observation
        )

        processed = preprocess_frame(max_observation)

        self.frames.append(processed)

        self.previous_observation = observation

        return stack_frames(list(self.frames))