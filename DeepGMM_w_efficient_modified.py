import sys
import torch
import numpy as np
import torch.nn as nn
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import random

from environment import taxi
from oadam import OAdam
# from dataset import *
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter


class SASR_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# def calc_game_objective(w, f, s, sprime, eta_s_a):
#     # calculate the tuple of objective functions that the g and f networks
#     # respectively are minimizing
#     # eta_s_a is eta[s,a] slicing.
#     w_of_s = torch.squeeze(w(s))
#     w_of_sprime = torch.squeeze(w(sprime))
#     f_of_sprime = torch.squeeze(f(sprime))
#     eta_s_a = torch.squeeze(eta_s_a)  # all 4 vectors shape=[batch_size]

#     epsilon = w_of_s * eta_s_a - w_of_sprime
#     c = f.get_coef()
#     vector = (f_of_sprime * epsilon
#               + c[0] * (w_of_s - 1.0)
#               + c[1] * (w_of_sprime - 1.0))
#     moment = vector.mean()
#     f_reg = (vector ** 2).mean()
#     return moment, -moment + 0.25 * f_reg

    
def calc_game_objective(w, f, s, sprime, eta_s_a, coeff=3.0):
    # calculate the tuple of objective functions that the g and f networks
    # respectively are minimizing
    # eta_s_a is eta[s,a] slicing.
    w_of_s = torch.squeeze(w(s))
    w_of_sprime= torch.squeeze(w(sprime))  
    f_of_sprime = torch.squeeze(f(sprime))
    eta_s_a = torch.squeeze(eta_s_a) #all 4 vectors shape=[batch_size]
    
    epsilon = w_of_s*eta_s_a-w_of_sprime
    vector = f_of_sprime*epsilon
    moment= vector.mean()
    f_reg =(vector**2).mean()
    
    constraint = ((w(s)).mean()-1)**2
    return moment+coeff*constraint, -moment + 0.25*f_reg



def on_policy(SASR, gamma):
    total_reward = 0.0
    discounted_t = 1.0
    self_normalizer = 0.0
    for (state, action, next_state, reward) in SASR:
        total_reward += reward * discounted_t
        self_normalizer += discounted_t
        discounted_t *= gamma
    return total_reward / self_normalizer  # why take the mean?


def train(train_loader, w, f, eta, w_optimizer, f_optimizer):
    for SASR in train_loader:
        s, a, sprime, r = SASR
        w_obj, f_obj = calc_game_objective(w, f, s, sprime,
                                           eta[s, a].unsqueeze(1))
        # print(w_obj, constraint)
        # do single first order optimization step on f and g
        w_optimizer.zero_grad()
        w_obj.backward(retain_graph=True)
        w_optimizer.step()

        f_optimizer.zero_grad()
        f_obj.backward()
        f_optimizer.step()


def validate(val_loader, w, f, gamma, eta, gt_reward, args, gt_w):
    dev_obj = []
    total_reward = []
    total_reward_gt = []

    for SASR in val_loader:
        s, a, sprime, r = SASR

        w_obj, f_obj = calc_game_objective(w, f, s, sprime,
                                           eta[s, a].unsqueeze(1))
        dev_obj.append(w_obj)
        if len(total_reward) == 0:
            total_reward = w(s).squeeze() * eta[s, a] * r.float()
            total_reward_gt = gt_w[s].squeeze() * eta[s, a] * r.float()
        else:
            total_reward = torch.cat(
                (total_reward, w(s).squeeze() * eta[s, a] * r.float()), dim=0)
            total_reward_gt = torch.cat(
                (total_reward_gt, gt_w[s].squeeze() * eta[s, a]
                 * r.float()), dim=0)
    total_reward = torch.cat((total_reward, w(s).squeeze() * eta[s, a]
                              * r.float()), dim=0)
    pred_reward = total_reward.mean()
    mean_obj = torch.Tensor(dev_obj).mean()
    print('pred_reward', float(pred_reward),
          'gt_reward', float(gt_reward),
          "mean_obj", float(mean_obj))
    mean_mse = (pred_reward - gt_reward) ** 2
    return mean_obj, mean_mse


def roll_out(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        for i_t in range(truncate_size):
            # env.render()
            p_action = policy[state, :]
            # action = np.random.choice(p_action.shape[0], 1, p=p_action)[0] Why change this line???
            action = np.random.choice(list(range(p_action.shape[0])),
                                      p=p_action)
            next_state, reward = env.step(action)

            SASR.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            # print env.state_decoding(state)
            # a = input()
            state = next_state
    mean_reward = total_reward / (num_trajectory * truncate_size)
    return SASR, frequency, mean_reward


class StateEmbedding(nn.Module):
    def __init__(self, num_state=2000, embedding_dim=1):
        super(StateEmbedding, self).__init__()
        self.embeddings = nn.Embedding(num_state, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)


class StateEmbeddingAdversary(nn.Module):
    def __init__(self, num_state=2000, embedding_dim=1):
        super(StateEmbeddingAdversary, self).__init__()
        self.embeddings = nn.Embedding(num_state, embedding_dim)
        self.c = nn.Parameter(torch.ones(2))

    def forward(self, inputs):
        return self.embeddings(inputs)

    def get_coef(self):
        return self.c


global_epoch = 0


def main():
    global global_epoch
    parser = argparse.ArgumentParser(description='taxi environment')
    parser.add_argument('--nt', type=int, required=False, default=1)
    parser.add_argument('--ts', type=int, required=False, default=100000)
    parser.add_argument('--gm', type=float, required=False, default=1.0)
    # parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--batch-size', default=1024, type=int)
    args = parser.parse_args()
    # args.val_writer = SummaryWriter(os.path.join(args.save_file, 'val'))

    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    print_freq = 10

    alpha = np.float(0.6)
    pi_eval = np.load('./taxi-policy/pi19.npy')
    # shape: (2000, 6)
    pi_behavior = np.load('./taxi-policy/pi3.npy')
    pi_behavior = alpha * pi_eval + (1 - alpha) * pi_behavior

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    #Why delete 200 and 300?
    SASR_b_train_raw, _, _ = roll_out(n_state, env, pi_behavior, args.nt,
                                      args.ts)
    SASR_b_val_raw, b_freq, _ = roll_out(n_state, env, pi_behavior, args.nt, args.ts)
    SASR_e, e_freq, rrr = roll_out(n_state, env, pi_eval, args.nt,
                              args.ts)  # pi_e doesn't need loader


    SASR_b_train = SASR_Dataset(SASR_b_train_raw)
    SASR_b_val = SASR_Dataset(SASR_b_val_raw)

    train_loader = torch.utils.data.DataLoader(
        SASR_b_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        SASR_b_val, batch_size=args.batch_size, shuffle=True)

    eta = torch.FloatTensor(pi_eval / pi_behavior)

    w = StateEmbedding()
    f = StateEmbeddingAdversary()

    for param in w.parameters():
        param.requires_grad = True
    for param in f.parameters():
        param.requires_grad = True
#     if torch.cuda.is_available():
#         w = w.cuda()
#         f = f.cuda()
#         eta = eta.cuda()
    w_optimizer = OAdam(w.parameters(), lr=1e-3, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=5e-3, betas=(0.5, 0.9))
    # w_optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     w_optimizer, patience=20, factor=0.5)
    # f_optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     f_optimizer, patience=20, factor=0.5)

    # SASR_b, b_freq, _ = roll_out(n_state, env, pi_behavior, 1, 10000000)
    # SASR_e, e_freq, _ = roll_out(n_state, env, pi_eval, 1, 10000000)
    # np.save('taxi7/taxi-policy/pi3_distrib.npy', b_freq/np.sum(b_freq))
    # np.save('taxi7/taxi-policy/pi19_distrib.npy', e_freq/np.sum(e_freq))

    b_freq_distrib = np.load('./pi3_distrib.npy')
    e_freq_distrib = np.load('./pi19_distrib.npy')
#     b_freq_distrib = b_freq/np.sum(b_freq)#np.load('pi3_distrib.npy')#
#     e_freq_distrib = e_freq/np.sum(e_freq)

    gt_w = torch.Tensor(e_freq_distrib / (1e-5+b_freq_distrib))

    # estimate policy value using ground truth w
    s = torch.LongTensor([s_ for s_, _, _, _ in SASR_b_train])
    a = torch.LongTensor([a_ for _, a_, _, _ in SASR_b_train])
    r = torch.FloatTensor([r_ for _, _, _, r_ in SASR_b_train])
    pi_b = torch.FloatTensor(pi_behavior)
    pi_e = torch.FloatTensor(pi_eval)
    gt_w_estimate = float((gt_w[s] * pi_e[s, a] / pi_b[s, a] * r).mean())
    print("estimate using ground truth w:", gt_w_estimate)

    # estimate policy value from evaluation policy roll out
    r_e = torch.FloatTensor([r_ for _, _, _, r_ in SASR_e])
    roll_out_estimate = float(r_e.mean())
    print("estimate using evaluation policy roll-out:", roll_out_estimate)

    s_all = torch.LongTensor(list(range(2000)))

    for epoch in range(5000):
        # print(epoch)
        global_epoch += 1
        dev_obj, dev_mse = validate(val_loader, w, f, args.gm, eta,
                                    roll_out_estimate, args, gt_w)
        if epoch % print_freq == 0:
            w_all = w(s_all).detach().flatten()
            w_rmse = float(((w_all - gt_w) ** 2).mean() ** 0.5)
            print("epoch %d, dev objective = %f, dev mse = %f, w rmse %f"
                  % (epoch, dev_obj, dev_mse, w_rmse))
            print("pred w:")
            print(w_all[:20])
            print("gt w:")
            print(gt_w[:20])
            print("mean w:", float(w(s).mean()))

            # torch.save({'w_model': w.state_dict(),
            #             'f_model': f.state_dict()},
            #            os.path.join(args.save_file, str(epoch) + '.pth'))
        train(train_loader, w, f, eta, w_optimizer, f_optimizer)
        # w_optimizer_scheduler.step(dev_mse)
        # f_optimizer_scheduler.step(dev_mse)


if __name__ == '__main__':
    main()
