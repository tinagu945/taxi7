import torch
import torch.nn as nn


class WAdversaryWrapper(nn.Module):
    def __init__(self, base_model, c=[1, 1]):
        super(WAdversaryWrapper, self).__init__()
        self.base_model = base_model
        self.c = nn.Parameter(torch.Tensor(c))

    def get_constraint_multipliers(self):
        return self.c

    def forward(self, inputs):
        return self.base_model(inputs)


class QAdversaryWrapper(nn.Module):
    def __init__(self, base_model, c=[1]):
        super(QAdversaryWrapper, self).__init__()
        self.base_model = base_model
        self.c = nn.Parameter(torch.Tensor(c))
        # self.c = torch.Tensor(c)
        # self.c.requies_grad = False

    def get_constraint_multipliers(self):
        return self.c

    def forward(self, inputs):
        return self.base_model(inputs)
