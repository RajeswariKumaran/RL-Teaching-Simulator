import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store one experience in the replay buffer.
        """
        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample experiences from the replay buffer.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)