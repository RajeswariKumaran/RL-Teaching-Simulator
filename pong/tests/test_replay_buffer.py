import numpy as np

from src.replay_buffer import ReplayBuffer


def test_replay_buffer_stores_experience():

    buffer = ReplayBuffer(capacity=10)

    state = np.zeros((4, 84, 84))
    next_state = np.ones((4, 84, 84))

    buffer.push(
        state,
        1,
        0.5,
        next_state,
        False
    )

    assert len(buffer) == 1


def test_replay_buffer_respects_capacity():

    buffer = ReplayBuffer(capacity=3)

    for i in range(5):

        state = np.zeros((4, 84, 84))

        buffer.push(
            state,
            i % 3,
            0.0,
            state,
            False
        )

    # Even though we added 5 experiences,
    # the buffer should keep only the most recent 3.
    assert len(buffer) == 3


def test_replay_buffer_samples_batch():

    buffer = ReplayBuffer(capacity=10)

    for i in range(5):

        state = np.zeros((4, 84, 84))

        buffer.push(
            state,
            i % 3,
            float(i),
            state,
            False
        )

    batch = buffer.sample(batch_size=3)

    assert len(batch) == 3