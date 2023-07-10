import torch

DQN_PARAMS = {
    "learning_rate": 2e-3,
    "hidden_dim": 128,
    "gamma": 0.98,
    "epsilon": 0.01,
    "target_update": 10,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
}

BUFFER_PARAMS = {
    "capacity": 10000,
}

TRAIN_PARAMS = {
    "minimal_size": 500,
    "batch_size": 64
}
