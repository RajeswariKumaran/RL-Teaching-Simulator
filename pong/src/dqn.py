import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, num_actions=3):
        super().__init__()

        self.network = nn.Sequential(

            # Input: (4, 84, 84)
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(3136, 512),
            nn.ReLU(),

            nn.Linear(512, num_actions)
        )

    def forward(self, state):
        return self.network(state)