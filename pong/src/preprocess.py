import numpy as np
from PIL import Image


def preprocess_frame(observation):
    """
    Convert an RGB Pong frame to grayscale
    and resize it to 84 x 84.
    """

    grayscale = np.mean(observation, axis=2)

    grayscale = grayscale.astype(np.uint8)

    resized = Image.fromarray(grayscale).resize(
        (84, 84),
        Image.Resampling.BILINEAR
    )

    return np.array(resized)