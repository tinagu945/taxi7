import torch.nn as nn


class TaxiSimpleCNN(nn.Module):
    """
    CNN model for taxi environment with tensor states
    states should be batch of states, each of shape (5, 5, 3)
    output should be
    """
    def __init__(self, num_out):
        w_0, c_0 = 5, 3
        c_1, k_1, p_1, s_1 = 10, 2, 0, 1
        c_2, k_2, p_2, s_2 = 20, 3, 0, 1
        w_1 = (w_0 + 2 * p_1 - k_1 + 1) // s_1
        w_2 = (w_1 + 2 * p_2 - k_2 + 1) // s_2
        super(TaxiSimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=c_0, out_channels=c_1,
                      kernel_size=k_1, stride=s_1, padding=p_1),
            nn.BatchNorm2d(c_1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c_1, out_channels=c_2,
                      kernel_size=k_2, stride=s_2, padding=p_2),
            nn.BatchNorm2d(c_2),
            nn.MaxPool2d(w_2))
        self.linear = nn.Sequential(
            nn.Linear(c_2, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            nn.Linear(20, num_out))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        return self.linear(x.squeeze())
