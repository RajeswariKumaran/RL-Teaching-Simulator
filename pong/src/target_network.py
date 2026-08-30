def update_target_network(model, target_model):
    target_model.load_state_dict(model.state_dict())