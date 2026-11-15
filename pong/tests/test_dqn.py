import torch

from src.dqn import DQN


def test_dqn_output_shape():

    model = DQN()

    # One fake Pong state.
    # 1 = batch size
    state = torch.zeros((1, 4, 84, 84))

    output = model(state)

    assert output.shape == (1, 3)