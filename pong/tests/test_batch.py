import numpy as np

from src.batch import prepare_batch


def test_prepare_batch_shapes():

    batch = []

    # Create 3 fake experiences
    for i in range(3):

        state = np.zeros((4, 84, 84), dtype=np.float32)
        next_state = np.ones((4, 84, 84), dtype=np.float32)

        experience = (
            state,
            i,
            float(i),
            next_state,
            False
        )

        batch.append(experience)

    states, actions, rewards, next_states, dones = prepare_batch(batch)

    assert states.shape == (3, 4, 84, 84)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4, 84, 84)
    assert dones.shape == (3,)