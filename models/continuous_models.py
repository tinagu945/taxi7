import torch.nn as nn
import torch

class QNetworkModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
        """Now the structure is very similar to the Q networks
        for pi for convenience, but can be changed arbitrarily."""
        super(QNetworkModel, self).__init__()
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim*2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim*2, out_dim)
                    )
        
    def forward(self, s):
        return self.model(s)
    