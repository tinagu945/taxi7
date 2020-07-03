import torch
import torch.nn as nn


class WAdversaryWrapper(nn.Module):
    def __init__(self, base_model):
        super(WAdversaryWrapper, self).__init__()
        self.base_model = base_model
        self.c = nn.Parameter(torch.ones(2))

    def get_constraint_multipliers(self):
        return self.c

    def forward(self, inputs):
        return self.base_model(inputs)
