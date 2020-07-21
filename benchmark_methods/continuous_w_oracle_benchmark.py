import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from models.continuous_models import QNetworkModel
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


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
    
    
def create_datasets(s_e, s_b, batch_size, train_ratio=0.8):
    # Input is S in SASR only
    train_len = int(train_ratio*s_e.size(0))
    
    train_s = torch.cat((s_e[:train_len], s_b[:train_len]))
    train_labels = torch.zeros((train_s.size(0),)).long()
    train_labels[:train_len] = 1
    train_data = TensorDataset(train_s, train_labels)   
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_s = torch.cat((s_e[train_len:], s_b[train_len:]))
    val_labels = torch.zeros((val_s.size(0),)).long()
    val_labels[:(s_e[train_len:].size(0))] = 1
    val_data = TensorDataset(val_s, val_labels)   
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    return train_data_loader, val_data_loader, val_s.size(0)


def calculate_continuous_w_oracle(env, pi_b, pi_e, gamma, hidden_dim, cuda=False, lr=1e-5,\
                               tau_len=1000000, burn_in=100000, batch_size=1024, epoch=100, \
                                  load = True, load_path=('tau_e_cartpole/','tau_b_cartpole/')):
    """
    :param env: environment (should be AbstractEnvironment)
    :param pi_b: behavior policy (should be from policies module)
    :param pi_e: evaluation policy (should be from policies module)
    :param gamma: discount factor
    :param num_s: number of different states
    :param tau_len: length to trajectory to use for monte-carlo estimate
    :param burn_in: burn-in period for monte-carlo sampling
    :return w: Network with same architecture as the trainining model
    """
#     tau_len=10
#     burn_in=1
    if load:
        s_e = torch.load(open(os.path.join(load_path[0], 's.pt'),'rb'))
        s_b= torch.load(open(os.path.join(load_path[1], 's.pt'),'rb'))
    else:
        tau_list_e = env.generate_roll_out(pi=pi_e, num_tau=1, tau_len=tau_len,
                                           gamma=gamma, burn_in=burn_in)
        tau_list_b = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
        tau_list_e.save(load_path[0])
        tau_list_b.save(load_path[1])
        s_e=tau_list_e[:][0]
        s_b=tau_list_b[:][0]
    
    w = WOracleModel(env.state_dim, 128, out_dim=2)
    train_data_loader, val_data_loader, val_len = create_datasets(s_e, s_b, batch_size, train_ratio=0.8)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(w.model.parameters()), lr=lr)
    
    lowest_loss = np.inf
    for i in range(epoch):
        print(i)
        w.train()
        for batch_idx, (data, labels) in enumerate(train_data_loader):
            if cuda:
                data, labels = data.cuda(), labels.cuda()
#             import pdb;pdb.set_trace()
            logits= w(data)   
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            
        if i %1==0:
            w.eval()
            all_correct=[]
            all_loss = []
            for batch_idx, (data, labels) in enumerate(val_data_loader):
                if cuda:
                    data, labels = data.cuda(), labels.cuda()
                
                logits= w(data)
                prob=logits.softmax(-1)
                loss = criterion(logits, labels)
                correct = (prob.argmax(-1)==labels).sum()
                all_correct.append(correct.item())
                all_loss.append(loss.item())
            print('test accuracy ', np.sum(all_correct)/val_len)
            avg_loss = np.mean(all_loss)
            print('avg loss ', avg_loss)
            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                torch.save(w.state_dict(), './continuous_w_oracle.pt')
                print('best model saved with loss ', lowest_loss)
    return w


def load_continuous_w_oracle(env, hidden_dim, path, cuda=False):
    w = WOracleModel(env.state_dim, 128, out_dim=2)
    w.load_state_dict(torch.load(open(path,'rb')))
    w.eval()
    if cuda:
        w = w.cuda()
    return w
    

def debug():
    from environments.cartpole_environment import CartpoleEnvironment
    from policies.mixture_policies import GenericMixturePolicy
    from policies.cartpole_policies import load_cartpole_policy
    
    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50

    env = CartpoleEnvironment()
    pi_e = load_cartpole_policy("cartpole_weights/cartpole_best.pt", temp, env.state_dim,
                                hidden_dim, env.num_a)
    pi_other = load_cartpole_policy("cartpole_weights/cartpole_210_318.0.pt", temp,
                                    env.state_dim, hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, alpha)
    
    calculate_continuous_w_oracle(env, pi_b, pi_e, gamma, hidden_dim)


if __name__ == "__main__":
    debug()
