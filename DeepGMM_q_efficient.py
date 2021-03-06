import sys
import torch
import numpy as np
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import random
import torch.nn as nn

from environment import taxi
from oadam import OAdam
from torch.utils.tensorboard import SummaryWriter


class SASR_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def calc_game_objective(q, f, s, a, sprime, r, pi_e, gamma):
    # calculate the tuple of objective functions that the g and f networks
    # respectively are minimizing
    # \sum_{a=1}^m \pi_e(a | S') q(S', a)
    c = pi_e.size(1)
#     bs = s.size(0)
    #select 2d-matrix(q,f) by row(s) and col(a) indices.
    q_of_s_a = torch.squeeze(q(s).gather(-1, a.unsqueeze(1).expand(-1, c))[:,0])
    f_of_s_a = torch.squeeze(f(s).gather(-1, a.unsqueeze(1).expand(-1, c))[:,0])
#     q_of_s_a = torch.zeros(bs)
#     f_of_s_a = torch.zeros(bs)
#     for i in range(bs):
#         q_of_s_a[i]=q(s[i])[a[i]]
#         f_of_s_a[i]=f(s[i])[a[i]]
#     import pdb;pdb.set_trace()
#     q_of_sprime_a = torch.squeeze(q(sprime).gather(-1, a.unsqueeze(1).expand(-1, c))[:,0])
    #bs x n_action
    E = (pi_e[sprime]*q(sprime)).sum(dim=1)
    #check r, f_of_s_a, q_of_s_a and E all same #of dim (1 or 2) True: all are [bs]
    eq = torch.squeeze(r + gamma*E - q_of_s_a)  # all 4 vectors shape=[batch_size]    
    
    vector = f_of_s_a * eq
    moment = vector.mean()
    f_reg = (vector ** 2).mean()
    return moment, -moment + 0.25 * f_reg


def on_policy(SASR, gamma):
    total_reward = 0.0
    self_normalizer = 0.0
    discounted_t = 1.0
    for state, action, next_state, reward in SASR:   
        total_reward += reward * discounted_t
        self_normalizer += discounted_t
        discounted_t *= gamma
    return total_reward / self_normalizer


def on_policy_old(SASR, gamma):
    total_reward = 0.0
    self_normalizer = 0.0
    for sasr in SASR:
        discounted_t = 1.0
        for state, action, next_state, reward in sasr:
            total_reward += reward * discounted_t
            self_normalizer += discounted_t
            discounted_t *= gamma
    return total_reward / self_normalizer

#direct estimator
def model_based(n_state, n_action, SASR, pi, gamma, nu0):
    R = np.load('gt_reward_table.npy')
    Q_table = np.zeros([n_state, n_action], dtype = np.float32)
    #TODO: use gt T: given s and a, prob(sprime)
    #use one SASR to estimate q and w, another SASR to calc policy based on q and w
    T= np.load('gt_T_pi19.npy') #pi_eval
    
#     T_est = np.zeros([n_state, n_action, n_state], dtype = np.float32)
#     for (state, action, next_state, reward) in SASR:
#         state = np.int(state)
#         action = np.int(action)
#         next_state = np.int(next_state)
#         T_est[state, action, next_state] += 1

#     T_est = T_est + 1e-9	# smoothing
#     T_est = T_est/np.sum(T_est, axis = -1)[:,:,None]
#     T += 1e-9
#     import pdb;pdb.set_trace()

    for i in range(100000):
        Q_table_new = np.zeros([n_state, n_action], dtype = np.float32)
        V_table = np.sum(Q_table*pi,1)
        for state in range(n_state):
            for action in range(n_action):
                Q_table_new[state,action] = R[state,action]+gamma*np.sum(T[state, action, :]*V_table)
        if (((Q_table_new-Q_table)**2).mean()** 0.5)<1e-7:
            print(i, 'Q_table converged.')
            break
        else:
            print(i,(((Q_table_new-Q_table)**2).mean()** 0.5))
        Q_table = np.copy(Q_table_new)
    return np.sum(np.sum(Q_table*pi,1).reshape(-1)*nu0)*(1-gamma), Q_table


def model_based_old(n_state, n_action, SASR, pi, gamma):
    R_my = np.load('gt_reward_table.npy')
    T_my = np.load('gt_T_pi19.npy') 
    Q_table = np.zeros([n_state, n_action], dtype = np.float32)
    Q_table_1 = np.zeros([n_state, n_action], dtype = np.float32)
    Q_table_2 = np.zeros([n_state, n_action], dtype = np.float32)
    Q_table_3 = np.zeros([n_state, n_action], dtype = np.float32)
    T = np.zeros([n_state, n_action, n_state], dtype = np.float32)
    R = np.zeros([n_state, n_action], dtype = np.float32)
    R_count = np.zeros([n_state, n_action], dtype = np.int32)
    for sasr in SASR:
        for state, action, next_state, reward in sasr:
            state = np.int(state)
            action = np.int(action)
            next_state = np.int(next_state)
            T[state, action, next_state] += 1
            R[state, action] += reward
            R_count[state, action] += 1
    d0 = np.ones([n_state, 1], dtype = np.float32)
    

    t = np.where(R_count > 0)
    t0 = np.where(R_count == 0)
    R[t] = R[t]/R_count[t]
    ###R[t0] = np.mean(R[t])
    T = T + 1e-9	# smoothing
    T = T/np.sum(T, axis = -1)[:,:,None]
    np.save('model_based_old_T.npy', T)
    np.save('model_based_old_R.npy', R)
    print('saved!')
    ddd = np.load("emp_hist.npy").reshape(-1)
    ####ddd = d0/np.sum(d0)
    for i in range(800):
        Q_table_new = np.zeros([n_state, n_action], dtype = np.float32)
        Q_table_new_1 = np.zeros([n_state, n_action], dtype = np.float32)
        Q_table_new_2 = np.zeros([n_state, n_action], dtype = np.float32)
        Q_table_new_3 = np.zeros([n_state, n_action], dtype = np.float32)
        V_table = np.sum(Q_table*pi,1)
        V_table_1 = np.sum(Q_table_1*pi,1)
        V_table_2 = np.sum(Q_table_2*pi,1)
        V_table_3 = np.sum(Q_table_3*pi,1)
        for state in range(n_state):
            for action in range(n_action):
                Q_table_new[state,action] = R[state,action]+gamma*np.sum(T[state, action, :]*V_table)
                Q_table_new_1[state,action] = R_my[state,action]+gamma*np.sum(T[state, action, :]*V_table_1)
                Q_table_new_2[state,action] = R_my[state,action]+gamma*np.sum(T_my[state, action, :]*V_table_2)
                Q_table_new_3[state,action] = R[state,action]+gamma*np.sum(T_my[state, action, :]*V_table_3)
  

        print(i,(((Q_table_new-Q_table)**2).mean()** 0.5))
        print((((Q_table_new_1-Q_table_1)**2).mean()** 0.5))
        print((((Q_table_new_2-Q_table_2)**2).mean()** 0.5))
        print((((Q_table_new_3-Q_table_3)**2).mean()** 0.5))
        Q_table = np.copy(Q_table_new)
        Q_table_1 = np.copy(Q_table_new_1)
        Q_table_2 = np.copy(Q_table_new_2)
        Q_table_3 = np.copy(Q_table_new_3)
    
    np.save('model_based_old_Q.npy', Q_table) 
    np.save('model_based_old_Q1.npy', Q_table_1)
    np.save('model_based_old_Q2.npy', Q_table_2)
    np.save('model_based_old_Q3.npy', Q_table_3)
    with open('find_stable.txt','a') as f:
        f.write(str(np.sum(np.sum(Q_table*pi,1).reshape(-1)*ddd)*(1-gamma))+
          str(np.sum(np.sum(Q_table_1*pi,1).reshape(-1)*ddd)*(1-gamma))+
         str(np.sum(np.sum(Q_table_2*pi,1).reshape(-1)*ddd)*(1-gamma))+
         str(np.sum(np.sum(Q_table_3*pi,1).reshape(-1)*ddd)*(1-gamma))+'\n')
        print(np.sum(np.sum(Q_table*pi,1).reshape(-1)*ddd)*(1-gamma),\
          np.sum(np.sum(Q_table_1*pi,1).reshape(-1)*ddd)*(1-gamma),\
         np.sum(np.sum(Q_table_2*pi,1).reshape(-1)*ddd)*(1-gamma),\
         np.sum(np.sum(Q_table_3*pi,1).reshape(-1)*ddd)*(1-gamma))
#     import pdb;pdb.set_trace()
    return np.sum(np.sum(Q_table_2*pi,1).reshape(-1)*ddd)*(1-gamma), Q_table_2


def train(train_loader, q, f, q_optimizer, f_optimizer, pi_e, gamma):
    for SASR in train_loader:
        s, a, sprime, r = SASR
        q_obj, f_obj = calc_game_objective(q, f, s, a, sprime, r, pi_e, gamma)
        # print(w_obj, constraint)
        
        f_optimizer.zero_grad()
        f_obj.backward(retain_graph=True)
        f_optimizer.step()

        q_optimizer.zero_grad()
        q_obj.backward()
        q_optimizer.step()

        

def validate(val_loader, q, f, roll_out_estimate, args, pi_e, gamma, est_model_based, gt_q_table, nu0):
    dev_obj = []
    
    for SASR in val_loader:
        s, a, sprime, r = SASR
        q_obj, f_obj = calc_game_objective(q, f, s, a, sprime, r, pi_e, gamma)       
        dev_obj.append(q_obj)
        
#     import pdb;pdb.set_trace()
    tmp = (q.embeddings.weight * pi_e).detach().numpy()
    pred_reward = np.sum(np.sum(tmp,1).reshape(-1)*nu0)*(1-gamma)
    mean_obj = torch.Tensor(dev_obj).mean()
    print('pred_reward', float(pred_reward),
          'roll_out_estimate', float(roll_out_estimate),
          'est_model_based', float(est_model_based),
          "mean_obj", float(mean_obj))
    mean_mse = (pred_reward - roll_out_estimate) ** 2

    if global_epoch % args.print_freq == 0:
        #weighted action prob
        q_rmse = float(((q.embeddings.weight.detach().numpy()[s] - gt_q_table[s]) ** 2).mean() ** 0.5)
        print("epoch %d, dev objective = %f, pred reward mse = %f, q rmse %f"
              % (global_epoch, mean_obj, mean_mse, q_rmse))
        print("pred q:")
        print(q.embeddings.weight[:5])
        print("gt q:")
        print(gt_q_table[:5])
        args.val_writer.add_scalar('q_rmse', q_rmse, global_epoch) 
        
    args.val_writer.add_scalar('mean_obj', mean_obj, global_epoch)
    args.val_writer.add_scalar('pred_reward(mean)_mse', mean_mse, global_epoch) 
    
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
            # action = np.random.choice(p_action.shape[0], 1, p=p_action)[0]
            action = np.random.choice(list(range(p_action.shape[0])), p=p_action)
            next_state, reward = env.step(action)
#             next_state, reward, done, _ = env.step(action)
#             if done:
#                 state = env.reset()

            SASR.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            # print env.state_decoding(state)
            # a = input()
            state = next_state
    mean_reward = total_reward / (num_trajectory * truncate_size)
    return SASR, frequency, mean_reward


def roll_out_old(state_num, env, policy, num_trajectory, truncate_size):
    SASR = []
    total_reward = 0.0
    frequency = np.zeros(state_num)
    for i_trajectory in range(num_trajectory):
        state = env.reset()
        sasr = []
        for i_t in range(truncate_size):
            #env.render()
            p_action = policy[state, :]
            action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
            next_state, reward = env.step(action)

            sasr.append((state, action, next_state, reward))
            frequency[state] += 1
            total_reward += reward
            #print env.state_decoding(state)
            #a = input()

            state = next_state
        SASR.append(sasr)
    return SASR, frequency, total_reward/(num_trajectory * truncate_size)


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
    parser.add_argument('--nt', type=int, required=False, default=1000)
    parser.add_argument('--ts', type=int, required=False, default=1000)
    parser.add_argument('--gm', type=float, required=False, default=0.98)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--batch-size', default=1024, type=int)
    args = parser.parse_args()
    args.val_writer = SummaryWriter(args.save_file)

    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    args.print_freq = 10
    gamma=args.gm

    alpha = np.float(0.6)
    pi_eval = np.load('taxi-policy/pi19.npy')
    # shape: (2000, 6)
    pi_behavior = np.load('taxi-policy/pi3.npy')
    pi_behavior = alpha * pi_eval + (1 - alpha) * pi_behavior

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    SASR_b_train_raw, _, _ = roll_out(n_state, env, pi_behavior, args.nt,
                                      args.ts)
    SASR_b_val_raw, _, _ = roll_out(n_state, env, pi_behavior, args.nt, args.ts)
    SASR_e, _, rrr = roll_out(n_state, env, pi_eval, args.nt,
                              args.ts)  # pi_e doesn't need loader
    # rrr (reward) is -0.1495

    SASR_b_train = SASR_Dataset(SASR_b_train_raw)
    SASR_b_val = SASR_Dataset(SASR_b_val_raw)

    train_loader = torch.utils.data.DataLoader(
        SASR_b_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        SASR_b_val, batch_size=args.batch_size, shuffle=True)

#     eta = torch.FloatTensor(pi_eval / pi_behavior)

    q = StateEmbedding(embedding_dim=n_action)
    f = StateEmbeddingAdversary(embedding_dim=n_action)

    for param in q.parameters():
        param.requires_grad = True
    for param in f.parameters():
        param.requires_grad = True
#     if torch.cuda.is_available():
#         w = w.cuda()
#         f = f.cuda()
#         eta = eta.cuda()

    q_optimizer = OAdam(q.parameters(), lr=1e-5, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=5e-5, betas=(0.5, 0.9))
#     q_optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         q_optimizer, patience=40, factor=0.5)
#     f_optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         f_optimizer, patience=40, factor=0.5)

    # SASR_b, b_freq, _ = roll_out(n_state, env, pi_behavior, 1, 10000000)
    # SASR_e, e_freq, _ = roll_out(n_state, env, pi_eval, 1, 10000000)
    # np.save('taxi7/taxi-policy/pi3_distrib.npy', b_freq/np.sum(b_freq))
    # np.save('taxi7/taxi-policy/pi19_distrib.npy', e_freq/np.sum(e_freq))

#     b_freq_distrib = np.load('pi3_distrib.npy')
#     e_freq_distrib = np.load('pi19_distrib.npy')

#     gt_w = torch.Tensor(e_freq_distrib / b_freq_distrib)

#     # estimate policy value using ground truth w
#     s = torch.LongTensor([s_ for s_, _, _, _ in SASR_b_train])
#     a = torch.LongTensor([a_ for _, a_, _, _ in SASR_b_train])
#     r = torch.FloatTensor([r_ for _, _, _, r_ in SASR_b_train])

    #TODO: get roll_out_estimate and est_model_based very close; if stuck, try all_compare.py
    pi_b = torch.FloatTensor(pi_behavior)
    pi_e = torch.FloatTensor(pi_eval)

#     gt_w_estimate = float((gt_w[s] * pi_e[s, a] / pi_b[s, a] * r).mean())
#     print("estimate using ground truth w:", gt_w_estimate)

    # estimate policy value from evaluation policy roll out
#     r_e = torch.FloatTensor([r_ for _, _, _, r_ in SASR_e])
#     roll_out_estimate = float(r_e.mean())
    roll_out_estimate = -0.26128581 #on_policy(SASR_e, gamma)
    print("estimate using evaluation policy roll-out:", roll_out_estimate)

    nu0=np.load("emp_hist.npy").reshape(-1)
    est_model_based, gt_q_table = model_based(n_state, n_action, SASR_e, pi_eval, gamma, nu0)
    print("Reward estimate using model based gt_q_table: ",est_model_based) 
#     old_est_model_based, old_gt_q_table = model_based_old(n_state, n_action, SASR_e, pi_eval, gamma)
#     print("Reward estimate using model based old gt_q_table: ",old_est_model_based)
#     import pdb;pdb.set_trace()
    
    for epoch in range(5000):
        # print(epoch)
        dev_obj, pred_reward_mse = validate(val_loader, q, f, roll_out_estimate, args, pi_e, gamma, \
                                    est_model_based, gt_q_table, nu0)
            # torch.save({'w_model': w.state_dict(),
            #             'f_model': f.state_dict()},
            #            os.path.join(args.save_file, str(epoch) + '.pth'))
        train(train_loader, q, f, q_optimizer, f_optimizer, pi_e, gamma)
#         q_optimizer_scheduler.step(pred_reward_mse)
#         f_optimizer_scheduler.step(pred_reward_mse)
        global_epoch += 1


def find_stable():
    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action
    pi_eval = np.load('taxi-policy/pi19.npy')
    gamma=1.0
    
    for i in [200,400,600,800,1000]:
        for j in range(10): 
            SASR_e, _, rrr = roll_out_old(n_state, env, pi_eval, i, 1000)
            roll_out_estimate = on_policy_old(SASR_e, gamma)
            print(i, "estimate using evaluation policy roll-out:", roll_out_estimate)
            with open('find_stable.txt','a') as f:
                f.write(str(i)+' '+str(j)+' '+str(roll_out_estimate)+'\n')
#             old_est_model_based, old_gt_q_table = model_based_old(n_state, n_action, SASR_e, pi_eval, gamma)
#             print(i, "estimate using evaluation policy roll-out:", roll_out_estimate)
#             print("Reward estimate using model based old gt_q_table: ",old_est_model_based)
    
    
    
if __name__ == '__main__':
#     main()
    find_stable()
