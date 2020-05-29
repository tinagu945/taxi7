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
from torch.utils.tensorboard import SummaryWriter


class SASR_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
    
def calc_game_objective(w, f, s, sprime, eta_s_a):
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
    return moment, -moment + 0.25*f_reg


def lagrange_constraint(coeff, w, s):
    constraint = ((w(s)).mean()-1)**2
    return coeff*constraint

#If neccessary, calc_obj_safe could be added with little modification

def get_eta(pi_eval, pi_behavior, n_state, n_action):
    #eq(6) in Liu's paper
    eta=torch.zeros((n_state, n_action))
    for state in range(n_state):
        for action in range(n_action):
            eta[state, action] = pi_eval[state, action]/pi_behavior[state, action]  
    return eta


def on_policy(SASR, gamma):
    total_reward = 0.0
    discounted_t = 1.0
    self_normalizer = 0.0
    for (state, action, next_state, reward) in SASR:
        total_reward += reward * discounted_t
        self_normalizer += discounted_t
        discounted_t *= gamma
    return total_reward/self_normalizer #why take the mean?


# def off_policy_evaluation_density_ratio(SASR, policy0, policy1, w, gamma):
#     #0 is behavior, 1 is eval
#     #eq(6) in Liu's paper
#     total_reward = 0.0
#     self_normalizer = 0.0
#     for sasr in SASR:
#         discounted_t = gamma
#         for state, action, next_state, reward in sasr:
#             policy_ratio = policy1[state, action]/policy0[state, action]
#             import pdb;pdb.set_trace()
#             total_reward += w(state).cpu().numpy() * policy_ratio * reward #### * discounted_t
#             self_normalizer += 1.0###density_ratio[state] * policy_ratio ######* discounted_t
#             #######discounted_t = gamma
#     return total_reward / self_normalizer


#TODO: should use dataloader to sample batches. The current way in below and calc_mse_safe is not standard
def train(train_loader, w, f, eta, coeff, w_optimizer, f_optimizer):
    # train our g function using DeepGMM algorithm

#     num_data = x_train.shape[0]

#     # decide random order of data for batches in this epoch
#     num_batch = math.ceil(num_data / batch_size)
#     train_idx = list(range(num_data))
#     random.shuffle(train_idx)
#     idx_iter = itertools.cycle(train_idx)

    # loop through training data in batches
    for SASR in train_loader:
        s, a, sprime, r = SASR
        s, a, sprime, r = s.cuda(), a.cuda(), sprime.cuda(), r.cuda()   
        
        w_obj, f_obj = calc_game_objective(w, f, s, sprime, eta[s,a].unsqueeze(1))
        constraint = lagrange_constraint(coeff, w, s)
#         print(w_obj, constraint)
        w_obj_constraint = w_obj+constraint
        f_obj_constraint = f_obj #+constraint
        # do single first order optimization step on f and g
        w_optimizer.zero_grad()
        w_obj_constraint.backward(retain_graph=True)
        w_optimizer.step()

        f_optimizer.zero_grad()
        f_obj_constraint.backward()
        f_optimizer.step()

                

def validate(val_loader, w, f, gamma, eta, gt_reward, args, gt_w):
#     pred_reward = off_policy_evaluation_density_ratio(SASR, pi_behavior, pi_eval, w, gamma)
    dev_obj=[]
    mean_reward = 0.0
    total_reward=[]
    total_reward_gt=[]
    data_used = 0
    self_normalizer = 0.0
    discounted_t = gamma
    
    for SASR in val_loader:
        s, a, sprime, r = SASR
        batch_size=s.size(0)
        s, a, sprime, r = s.cuda(), a.cuda(), sprime.cuda(), r.cuda()  
        
        w_obj, f_obj = calc_game_objective(w, f, s, sprime, eta[s,a].unsqueeze(1))
        dev_obj.append(w_obj)
#         import pdb;pdb.set_trace()
#         policy_ratio = pi_eval[s,a]/pi_behavior[s,a]
        #running average
#         mean_reward = (mean_reward*data_used + (w(s).squeeze() * policy_ratio * r).sum())/(data_used+batch_size)
#         data_used+=batch_size
        if len(total_reward)==0:
            total_reward = w(s).squeeze() * eta[s,a] * r
#             import pdb;pdb.set_trace()
            total_reward_gt = gt_w[s].squeeze() * eta[s,a] * r
        else:
#             import pdb;pdb.set_trace()
            total_reward = torch.cat((total_reward, w(s).squeeze() * eta[s,a] * r), dim=0)
            total_reward_gt = torch.cat((total_reward_gt, gt_w[s].squeeze() * eta[s,a] * r), dim=0)
#         
    total_reward = torch.cat((total_reward, w(s).squeeze() * eta[s,a] * r), dim=0)
    pred_reward = total_reward.mean() 
    mean_obj = torch.Tensor(dev_obj).mean()
    print('pred_reward', pred_reward, 'gt_w estimate', total_reward_gt.mean(), 'gt_reward', gt_reward)
#     print(total_reward[:10], pred_reward, dev_obj[:10], mean_obj)
    mean_mse = (pred_reward-gt_reward)**2
    args.val_writer.add_scalar('dev_obj', mean_obj, global_epoch)
    args.val_writer.add_scalar('policy mse', mean_mse, global_epoch)
    return mean_obj, mean_mse


def roll_out(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        for i_t in range(truncate_size):
            #env.render()
            p_action = policy[state, :]
            action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
            next_state, reward = env.step(action)

            SASR.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            #print env.state_decoding(state)
            #a = input()
            state = next_state
    return SASR, frequency, total_reward/(num_trajectory * truncate_size) #SASR ntxts



    
class StateEmbedding(nn.Module):
    def __init__(self, num_state=2000, embedding_dim=1):
        super(StateEmbedding, self).__init__()
        self.embeddings = nn.Embedding(num_state, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)

global_epoch=0
def main():
    global global_epoch
    parser = argparse.ArgumentParser(description='taxi environment')
    parser.add_argument('--nt', type = int, required = False, default = 200)
    parser.add_argument('--ts', type = int, required = False, default = 400)
    parser.add_argument('--gm', type = float, required = False, default = 1.0)
    parser.add_argument('--save_file', type = str, required = True)
    parser.add_argument('--constraint_alpha', type = float, required = False, default = 0.3)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()
    args.val_writer = SummaryWriter(os.path.join(args.save_file, 'val'))


    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    print_freq=20

    alpha = np.float(0.6)
    pi_eval = np.load('taxi-policy/pi19.npy')
    pi_behavior = np.load('taxi-policy/pi3.npy')
    pi_behavior = alpha * pi_eval + (1-alpha) * pi_behavior
    
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    SASR_b_train_raw, _, _ = roll_out(n_state, env, pi_behavior, args.nt, args.ts)
    random.seed(200)
    np.random.seed(200)
    torch.manual_seed(200)
    SASR_b_val_raw, _, _ = roll_out(n_state, env, pi_behavior, args.nt, args.ts)
    random.seed(300)
    np.random.seed(300)
    torch.manual_seed(300)
    SASR_e, _, rrr = roll_out(n_state, env, pi_eval, args.nt, args.ts) #pi_e doesn't need loader
    #rrr (reward) is -0.1495
    
    SASR_b_train = SASR_Dataset(SASR_b_train_raw)
    SASR_b_val = SASR_Dataset(SASR_b_val_raw)  
    
    train_loader = torch.utils.data.DataLoader(
        SASR_b_train, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        SASR_b_val, batch_size=args.batch_size, shuffle=False)
    
    eta = get_eta(pi_eval, pi_behavior, n_state, n_action)
    gt_reward = on_policy(np.array(SASR_e), args.gm)
    print("HERE000000000 estimate using evaluation policy roll-out:", gt_reward)
    eta = torch.FloatTensor(pi_eval / pi_behavior).cuda()
    r_e = torch.FloatTensor([r_ for _, _, _, r_ in SASR_e]).cuda()
    gt_reward = float(r_e.mean())
    print("HERE estimate using evaluation policy roll-out:", gt_reward)

    
    
    w = StateEmbedding()
    f = StateEmbedding()
    for param in w.parameters():
        param.requires_grad = True
    for param in f.parameters():
        param.requires_grad = True
    if torch.cuda.is_available():
        w = w.cuda()
        f = f.cuda()
        eta = eta.cuda()
    w_optimizer = OAdam(w.parameters(), lr=2e-6, betas=(0.5, 0.9)) 
    f_optimizer = OAdam(f.parameters(), lr=1e-6, betas=(0.5, 0.9)) 
    w_optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(w_optimizer, patience=20, factor=0.5)
    f_optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(f_optimizer, patience=20, factor=0.5)

    
    b_freq_distrib = np.load('pi3_distrib.npy')
    e_freq_distrib = np.load('pi19_distrib.npy')
    gt_w = torch.Tensor(e_freq_distrib/(b_freq_distrib)).cuda()  #for numerical stability
    s=torch.linspace(0, 1999, steps=2000).cuda().long()
    
    
    s = torch.LongTensor([s_ for s_, _, _, _ in SASR_b_train]).cuda()
    a = torch.LongTensor([a_ for _, a_, _, _ in SASR_b_train]).cuda()
    r = torch.FloatTensor([r_ for _, _, _, r_ in SASR_b_train]).cuda()
    pi_b = torch.FloatTensor(pi_behavior).cuda()
    pi_e = torch.FloatTensor(pi_eval).cuda()
    gt_w_estimate = float((gt_w[s] * pi_e[s, a] / pi_b[s, a] * r).mean())
    print("HERE estimate using ground truth w:", gt_w_estimate)

    for epoch in range(5000):
#         print(epoch)
        global_epoch += 1
        dev_obj, dev_mse = validate(val_loader, w, f, args.gm, eta, gt_reward, args, gt_w)
        if epoch % print_freq == 0:
            print("epoch %d, dev objective = %f, dev mse = %f"
                      % (epoch, dev_obj, dev_mse))
            print(w(s).flatten()[:20],'\n', gt_w[:20], ' \n mean',  (w(s)).mean())
            
            torch.save({'w_model': w.state_dict(),
                       'f_model': f.state_dict()}, 
                       os.path.join(args.save_file, str(epoch)+'.pth'))
        train(train_loader, w, f, eta, args.constraint_alpha, w_optimizer, f_optimizer)  
        w_optimizer_scheduler.step(dev_mse)
        f_optimizer_scheduler.step(dev_mse)
    
if __name__=='__main__':
    main()