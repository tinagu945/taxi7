import os
import torch
from torch.utils.data import Dataset, DataLoader


class TauListDataset(Dataset):
    def __init__(self, tau_list):
        # Destroy the num_tau structure
        self.s = torch.cat([s for s, _, _, _ in tau_list])
        self.a = torch.cat([a for _, a, _, _ in tau_list])
        self.s_prime = torch.cat([s for _, _, s, _ in tau_list])
        self.r = torch.cat([r for _, _, _, r in tau_list])
        self.length = len(self.s)
        self.lens = torch.LongTensor([len(s_) for s_, _, _, _ in tau_list])

        assert len(self.a) == self.length
        assert len(self.s_prime) == self.length
        assert len(self.r) == self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.s_prime[idx], self.r[idx]

    def get_data_loader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def save(self, save_path, prefix=""):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.s, os.path.join(save_path, prefix+"s.pt"))
        torch.save(self.a, os.path.join(save_path, prefix+"a.pt"))
        torch.save(self.s_prime, os.path.join(save_path, prefix+"s_prime.pt"))
        torch.save(self.r, os.path.join(save_path, prefix+"r.pt"))
        torch.save(self.lens, os.path.join(save_path, prefix+"lens.pt"))

    @staticmethod
    def load(save_path, prefix=""):
        s = torch.load(os.path.join(save_path, prefix+"s.pt"))
        a = torch.load(os.path.join(save_path, prefix+"a.pt"))
        s_prime = torch.load(os.path.join(save_path, prefix+"s_prime.pt"))
        r = torch.load(os.path.join(save_path, prefix+"r.pt"))
        lens = torch.load(os.path.join(save_path, prefix+"lens.pt"))
        tau_list = []
        s_i = 0
        for l in lens:
            e_i = s_i + int(l)
            tau_list.append((s[s_i:e_i], a[s_i:e_i],
                             s_prime[s_i:e_i], r[s_i:e_i]))
            s_i = e_i
        return TauListDataset(tau_list)

    def restore_strcture(self, discrete=False):
        intervals = [0]
        for i in self.lens:
            intervals.append(intervals[-1]+int(i))
        SASR = []
        for i in range(1, len(intervals)):
            discounted_t = 1.0
            if len(self.s.size()) == 2:
                a = self.a.unsqueeze(-1)
                r = self.r.unsqueeze(-1)
                s = self.s
                s_prime = self.s_prime
            elif len(self.s.size()) == 1:
                a = self.a.unsqueeze(-1)
                r = self.r.unsqueeze(-1)
                s = self.s.unsqueeze(-1)
                s_prime = self.s_prime.unsqueeze(-1)
            if discrete:
                s, a, s_prime, r = s.long(), a.long(), s_prime.long(), r.long()
            else:
                s, a, s_prime, r = s.float(), a.float(), s_prime.float(), r.float()
            sasr = torch.cat([s[intervals[i-1]:intervals[i]], a[intervals[i-1]:intervals[i]],
                              s_prime[intervals[i-1]:intervals[i]], r[intervals[i-1]:intervals[i]]], dim=-1)

            SASR.append(sasr)
        return SASR


# def restore_dataset_from_load(path, prefix):
#     train_s = torch.load(open(os.path.join(path, prefix+'s.pt'),'rb')).unsqueeze(1)
#     train_a = torch.load(open(os.path.join(path, prefix+'a.pt'),'rb')).unsqueeze(-1)
#     train_s_prime = torch.load(open(os.path.join(path, prefix+'s_prime.pt'),'rb')).unsqueeze(1)
#     train_r = torch.load(open(os.path.join(path, prefix+'r.pt'),'rb')).unsqueeze(-1)

#     return train_data
