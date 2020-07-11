import torch.nn as nn


class QNetworkModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim*2),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim*2, action_dim)
                    )
        
    def forward(self, s):
        return self.model(s)