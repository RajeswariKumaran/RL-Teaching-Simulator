import numpy as np


def stack_frames(frames):
    """
    Stack four consecutive processed Pong frames.
    """

    return np.stack(frames, axis=0)