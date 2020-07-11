import numpy as np
import torch
from policies.continuous_policy import ContinuousPolicy

def load_cartpole_policy(path, temp, state_dim, action_dim, hidden_dim):
    # The Q network
    model = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim*2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim*2, action_dim)
                )
    model.load_state_dict(torch.load(path))
    model.eval()
    return ContinuousPolicy(model, temp)