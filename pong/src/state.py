from collections import deque

from src.frame_stack import stack_frames
from src.preprocess import preprocess_frame


class PongState:

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, observation):

        processed = preprocess_frame(observation)

        self.frames.clear()

        for _ in range(self.num_frames):
            self.frames.append(processed)

        return stack_frames(list(self.frames))

    def step(self, observation):

        processed = preprocess_frame(observation)

        self.frames.append(processed)

        return stack_frames(list(self.frames))