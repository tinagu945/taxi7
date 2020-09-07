import numpy as np
import torch
from policies.continuous_policy import QNetworkPolicy
from models.continuous_models import QOracleModel


def load_cartpole_policy(path, temp, state_dim, hidden_dim, action_dim):
    # The Q network
    model = QOracleModel(state_dim, hidden_dim, out_dim=action_dim)
    model.model.load_state_dict(torch.load(path))
    model.model.eval()
    return QNetworkPolicy(model, temp)
