import torch.nn as nn
import torch


class WNetworkModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1, positive_output=False):
        """Now the structure is same as the QNetworkModel for convenience, but can be changed arbitrarily."""
        super(WNetworkModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, out_dim),
        )
        self.square_output = positive_output

    def forward(self, s):
        if self.square_output:
            return torch.abs(self.model(s))
        else:
            return self.model(s)


class StateClassifierModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
        """Now the structure is very similar to the Q networks
        for pi for convenience, but can be changed arbitrarily."""
        super(StateClassifierModel, self).__init__()
        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(state_dim, hidden_dim),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(hidden_dim, hidden_dim*2),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(hidden_dim*2, hidden_dim*2),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(hidden_dim*2, out_dim)
        # )
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim*2, out_dim),
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


class WOracleModel(nn.Module):
    def __init__(self, state_classifier, train_data_loader, reg=1e-7):
        nn.Module.__init__(self)
        self.reg = reg
        self.state_classifier = state_classifier
        w_sum = 0
        w_norm = 0
        for s, _ in train_data_loader:
            w = self._calc_w_raw(s)
            w_sum += float(w.sum().detach())
            w_norm += len(w)
        self.w_mean = w_sum / w_norm

    def _calc_w_raw(self, s):
        b_prob = self.state_classifier(s).softmax(-1)[:, 0]
        w = (1 - b_prob) / (b_prob + 1e-7)
        return w

    def forward(self, s):
        w_raw = self._calc_w_raw(s)
        # return w_raw / self.w_mean
        return w_raw


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


class QNetworkModelSimple(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1, neg_output=False):
        super(QNetworkModelSimple, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(hidden_dim*2),
            # torch.nn.Linear(hidden_dim*2, hidden_dim*4),
            # torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(hidden_dim*4),
            torch.nn.Linear(hidden_dim*2, out_dim),
            # torch.nn.BatchNorm1d(out_dim),
        )
        self.neg_output = neg_output


    def forward(self, s):
        if self.neg_output:
            return -torch.abs(self.model(s))
        else:
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
