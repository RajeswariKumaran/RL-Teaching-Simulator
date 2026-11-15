import torch

from src.dqn import DQN


def main():

    model = DQN()

    # One fake Pong state
    state = torch.zeros((1, 4, 84, 84))

    q_values = model(state)

    print("Q-values:")
    print(q_values)

    action = torch.argmax(q_values, dim=1)

    print("\nSelected action:")
    print(action.item())


if __name__ == "__main__":
    main()