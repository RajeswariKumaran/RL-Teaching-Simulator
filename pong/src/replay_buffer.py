import random
import numpy as np


class ReplayBuffer:

    def __init__(self, capacity=50_000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = []

        self.position = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        # New experiences get the highest current priority.
        max_priority = (
            max(self.priorities)
            if self.priorities
            else 1.0
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)

        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority

        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(
        self,
        batch_size,
        beta=0.4
    ):
        priorities = np.array(
            self.priorities,
            dtype=np.float32
        )

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probabilities
        )

        experiences = [
            self.buffer[index]
            for index in indices
        ]

        # Importance-sampling weights.
        weights = (
            len(self.buffer) * probabilities[indices]
        ) ** (-beta)

        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(
        self,
        indices,
        priorities
    ):
        for index, priority in zip(
            indices,
            priorities
        ):
            self.priorities[index] = (
                float(priority) + 1e-6
            )

    def __len__(self):
        return len(self.buffer)