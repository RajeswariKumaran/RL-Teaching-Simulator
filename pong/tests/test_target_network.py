import torch

from src.dqn import DQN
from src.target_network import update_target_network


def test_update_target_network_copies_policy_weights():

    model = DQN()
    target_model = DQN()

    # Make the policy model deliberately different
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    update_target_network(model, target_model)

    # Every parameter should now match
    for model_parameter, target_parameter in zip(
        model.parameters(),
        target_model.parameters(),
    ):
        assert torch.equal(model_parameter, target_parameter)