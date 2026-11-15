import matplotlib.pyplot as plt

from src.pong_environment import PongEnvironment
from src.preprocess import preprocess_frame


def main():

    env = PongEnvironment()

    observation, info = env.reset()

    processed = preprocess_frame(observation)

    print("Original shape:", observation.shape)
    print("Processed shape:", processed.shape)

    plt.figure()
    plt.imshow(observation)
    plt.title("Original Pong Frame")
    plt.axis("off")

    plt.figure()
    plt.imshow(processed, cmap="gray")
    plt.title("Grayscale Pong Frame")
    plt.axis("off")

    plt.show()

    env.close()


if __name__ == "__main__":
    main()