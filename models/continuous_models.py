import torch.nn as nn
import torch


class WNetworkModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
        """Now the structure is same as the QNetworkModel for convenience, but can be changed arbitrarily."""
        super(WNetworkModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, out_dim)
        )

    def forward(self, s):
        return self.model(s)


class WOracleModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
        """Now the structure is very similar to the Q networks
        for pi for convenience, but can be changed arbitrarily."""
        super(WOracleModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, out_dim)
        )

    def forward(self, s):
        return self.model(s)

    @staticmethod
    def load_continuous_w_oracle(env, hidden_dim, path, cuda=False):
        w = WOracleModel(env.state_dim, 128, out_dim=2)
        w.load_state_dict(torch.load(open(path, 'rb')))
        w.eval()
        if cuda:
            w = w.cuda()
        return w


class QNetworkModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
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

    @staticmethod
    def load_continuous_q(env, hidden_dim, out_dim, path, cuda=False):
        q = QNetworkModel(env.state_dim, hidden_dim, out_dim=out_dim)
        q.model.load_state_dict(torch.load(open(path, 'rb')))
        # q.model.eval()
        if cuda:
            q.model = q.model.cuda()
        return q


class QOracleModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
        super(QOracleModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, out_dim)
        )

    def forward(self, s):
        return self.model(s)

    @staticmethod
    def load_continuous_q_oracle(env, hidden_dim, out_dim, path, cuda=False):
        q = QOracleModel(env.state_dim, hidden_dim, out_dim=out_dim)
        q.model.load_state_dict(torch.load(open(path, 'rb')))
        q.model.eval()
        if cuda:
            q.model = q.model.cuda()
        return q
