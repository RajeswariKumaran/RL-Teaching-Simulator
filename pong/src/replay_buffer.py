import random

import numpy as np


class ReplayBuffer:
    """
    Memory-efficient Atari replay buffer.

    One 84x84 frame is stored for each environment step. Four-frame
    states are reconstructed when sampled. The next frame is also kept
    for every transition so the newest transition can be sampled without
    waiting for another environment step. This also keeps the buffer
    compatible with small unit tests that insert only a few transitions.
    """

    def __init__(self, capacity=100_000, frame_shape=(84, 84)):
        self.capacity = capacity
        self.frame_shape = frame_shape

        self.frames = np.empty(
            (capacity, *frame_shape), dtype=np.uint8
        )
        self.next_frames = np.empty(
            (capacity, *frame_shape), dtype=np.uint8
        )
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.episode_starts = np.empty(capacity, dtype=np.bool_)
        self.sequence_ids = np.full(capacity, -1, dtype=np.int64)

        self.position = 0
        self.size = 0
        self.next_sequence_id = 0
        self._next_is_episode_start = True

    def push(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)

        expected_shape = (4, *self.frame_shape)
        if state.shape != expected_shape:
            raise ValueError(
                f"Expected state shape {expected_shape}, got {state.shape}"
            )
        if next_state.shape != expected_shape:
            raise ValueError(
                f"Expected next_state shape {expected_shape}, "
                f"got {next_state.shape}"
            )

        index = self.position

        self.frames[index] = state[-1]
        self.next_frames[index] = next_state[-1]
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.episode_starts[index] = self._next_is_episode_start
        self.sequence_ids[index] = self.next_sequence_id

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.next_sequence_id += 1
        self._next_is_episode_start = bool(done)

    def _index_for_sequence(self, sequence_id):
        index = sequence_id % self.capacity
        if self.sequence_ids[index] != sequence_id:
            return None
        return index

    def _build_state(self, sequence_id):
        """Reconstruct four frames ending at sequence_id."""
        current_index = self._index_for_sequence(sequence_id)
        if current_index is None:
            return None

        collected = []
        first_frame = self.frames[current_index]

        # Walk backwards so episode boundaries are handled correctly.
        for offset in range(4):
            wanted_sequence = sequence_id - offset
            index = self._index_for_sequence(wanted_sequence)

            if index is None:
                frame = first_frame
            else:
                frame = self.frames[index]
                if self.episode_starts[index]:
                    first_frame = frame

            collected.append(frame)

            if index is None or self.episode_starts[index]:
                break

        while len(collected) < 4:
            collected.append(first_frame)

        collected.reverse()
        return np.stack(collected, axis=0)

    def _valid_transition_ids(self):
        if self.size == 0:
            return []

        oldest = max(0, self.next_sequence_id - self.size)
        newest = self.next_sequence_id - 1

        valid = []
        for sequence_id in range(oldest, newest + 1):
            if self._index_for_sequence(sequence_id) is not None:
                valid.append(sequence_id)
        return valid

    def _build_transition(self, sequence_id):
        index = self._index_for_sequence(sequence_id)
        if index is None:
            raise RuntimeError("Sampled transition is no longer available.")

        state = self._build_state(sequence_id)
        if state is None:
            raise RuntimeError("Could not reconstruct state.")

        # For a normal transition, the next state's first three frames are
        # the last three frames of the current state, followed by the next
        # environment frame. For a terminal transition the stored next frame
        # is still used; the done flag prevents bootstrapping from it.
        next_frame = self.next_frames[index]
        next_state = np.concatenate(
            [state[1:], next_frame[None, ...]],
            axis=0,
        )

        return (
            state,
            int(self.actions[index]),
            float(self.rewards[index]),
            next_state,
            bool(self.dones[index]),
        )

    def sample(self, batch_size):
        valid_ids = self._valid_transition_ids()
        if len(valid_ids) < batch_size:
            raise ValueError(
                f"Only {len(valid_ids)} valid transitions are available; "
                f"cannot sample {batch_size}."
            )

        selected = random.sample(valid_ids, batch_size)
        return [self._build_transition(sequence_id) for sequence_id in selected]

    def __len__(self):
        return self.size
