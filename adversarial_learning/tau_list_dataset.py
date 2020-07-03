import torch
from torch.utils.data import Dataset, DataLoader


class TauListDataset(Dataset):
    def __init__(self, tau_list):
        self.s = torch.cat([s for s, _, _, _ in tau_list])
        self.a = torch.cat([a for _, a, _, _ in tau_list])
        self.s_prime = torch.cat([s for _, _, s, _ in tau_list])
        self.r = torch.cat([r for _, _, _, r in tau_list])
        self.length = len(self.s)
        assert len(self.a) == self.length
        assert len(self.s_prime) == self.length
        assert len(self.r) == self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.s_prime[idx], self.r[idx]


class TauListDataLoader(DataLoader):
    def __init__(self, tau_list, batch_size):
        dataset = TauListDataset(tau_list)
        DataLoader.__init__(self, dataset, batch_size=batch_size, shuffle=True)

