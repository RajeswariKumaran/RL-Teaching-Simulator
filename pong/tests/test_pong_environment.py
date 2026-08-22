import numpy as np

from src.preprocess import preprocess_frame
from src.pong_environment import PongEnvironment
from src.state import PongState


def test_environment_reset():

    env = PongEnvironment()

    observation, info = env.reset()

    assert observation is not None

    env.close()


def test_environment_action_space():

    env = PongEnvironment()

    assert len(env.actions) == 3

    assert env.actions[0] == 0
    assert env.actions[1] == 3
    assert env.actions[2] == 2

    env.close()


def test_environment_step():

    env = PongEnvironment()

    env.reset()

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation is not None

    env.close()



def test_preprocess_frame():

    # Create a fake RGB Pong frame
    observation = np.zeros((210, 160, 3), dtype=np.uint8)

    processed = preprocess_frame(observation)

    # The RGB channels should be removed
    # assert processed.shape == (210, 160)
    assert processed.shape == (84, 84)

    # It should still be an 8-bit image
    assert processed.dtype == np.uint8

from src.frame_stack import stack_frames


def test_stack_frames():

    frames = [
        np.zeros((84, 84), dtype=np.uint8)
        for _ in range(4)
    ]

    state = stack_frames(frames)

    assert state.shape == (4, 84, 84)
    assert state.dtype == np.uint8

from src.state import PongState


def test_pong_state_reset():

    observation = np.zeros(
        (210, 160, 3),
        dtype=np.uint8
    )

    state_manager = PongState()

    state = state_manager.reset(observation)

    assert state.shape == (4, 84, 84)
    assert state.dtype == np.uint8

def test_pong_state_step():

    observation1 = np.zeros(
        (210, 160, 3),
        dtype=np.uint8
    )

    observation2 = np.ones(
        (210, 160, 3),
        dtype=np.uint8
    )

    state_manager = PongState()

    state_manager.reset(observation1)

    state = state_manager.step(observation2)

    assert state.shape == (4, 84, 84)

    # The newest frame should be different
    assert not np.array_equal(state[0], state[-1])