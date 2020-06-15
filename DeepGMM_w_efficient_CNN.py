import sys
import torch
import numpy as np
import argparse
import os
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import random

from environment import taxi
from oadam import OAdam

from CNN_model import SimpleCNN 
from CNN_dataset_mat import SASR_Dataset
from torch.utils.tensorboard import SummaryWriter

def calc_game_objective(w, f, image_s, image_sprime, eta_s_a):
    # calculate the tuple of objective functions that the g and f networks
    # respectively are minimizing
    # eta_s_a is eta[s,a] slicing.
    w_of_s = torch.squeeze(w(image_s))
    w_of_sprime = torch.squeeze(w(image_sprime))
    f_of_sprime = torch.squeeze(f(image_sprime))
    eta_s_a = torch.squeeze(eta_s_a) # all 4 vectors shape=[batch_size]

    epsilon = w_of_s * eta_s_a - w_of_sprime
    c = f.get_coef()
    vector = (f_of_sprime * epsilon
              + c[0] * (w_of_s - 1.0)
              + c[1] * (w_of_sprime - 1.0))
    moment = vector.mean()
    f_reg = (vector ** 2).mean()
    return moment, -moment + 0.25 * f_reg


def on_policy(SASR, gamma):
    total_reward = 0.0
    discounted_t = 1.0
    self_normalizer = 0.0
    for (state, action, next_state, reward) in SASR:
        total_reward += reward * discounted_t
        self_normalizer += discounted_t
        discounted_t *= gamma
    return total_reward / self_normalizer  # why take the mean?


def train(loader, w, f, eta, w_optimizer, f_optimizer):
    i=0
    for image_s, image_sprime, SASR in loader:
#         print('train batch', i)
        image_s, image_sprime = image_s.cuda(), image_sprime.cuda()    
        s, a,= SASR[:,0].cuda(), SASR[:,1].cuda()
        w_obj, f_obj = calc_game_objective(w, f, image_s, image_sprime, 
                                           eta[s, a].unsqueeze(1))
        # print(w_obj, constraint)
        # do single first order optimization step on f and g
        w_optimizer.zero_grad()
        w_obj.backward(retain_graph=True)
        w_optimizer.step()
        
        f_optimizer.zero_grad()
        f_obj.backward()
        f_optimizer.step()

        i+=1


def validate(loader, w, f, gamma, eta, gt_reward, args, gt_w):
    dev_obj = []
    total_reward = []
    total_w = []
    for image_s, image_sprime, SASR in loader:
        image_s, image_sprime = image_s.cuda(), image_sprime.cuda()    
        s, a, r = SASR[:,0].cuda(), SASR[:,1].cuda(), SASR[:,3].cuda().float()
        w_obj, f_obj = calc_game_objective(w, f, image_s, image_sprime, 
                                           eta[s, a].unsqueeze(1))
        dev_obj.append(w_obj)
        if len(total_reward) == 0:
            total_reward = w(image_s).squeeze() * eta[s, a] * r
            total_w = w(image_s)
        else:
#             import pdb;pdb.set_trace()
            total_reward = torch.cat(
                (total_reward, w(image_s).squeeze() * eta[s, a] * r), dim=0)
            total_w = torch.cat((total_w, w(image_s)), dim=0)
            
    pred_reward = total_reward.mean().cpu()
    mean_obj = torch.Tensor(dev_obj).mean().cpu()
    print('pred_reward', float(pred_reward),
          'gt_reward', float(gt_reward),
          "mean_obj", float(mean_obj))
    mean_mse = (pred_reward - gt_reward) ** 2
    
    if global_epoch % args.print_freq == 0:
        total_w =  total_w.cpu()
        w_rmse = float(((total_w - gt_w) ** 2).mean() ** 0.5)
        print("epoch %d, dev objective = %f, dev mse = %f, w rmse %f"
              % (global_epoch, mean_obj, mean_mse, w_rmse))
        print("pred w:")
        print(total_w[:20])
        print("gt w:")
        print(gt_w[:20])
        print("mean w:", float(total_w.mean()))   
        del total_w
        args.val_writer.add_scalar('w_rmse', w_rmse, global_epoch) 
        
    args.val_writer.add_scalar('mean_obj', mean_obj, global_epoch)
    args.val_writer.add_scalar('mean_mse', mean_mse, global_epoch) 
    
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


global_epoch = 0


def main():
    global global_epoch
    parser = argparse.ArgumentParser(description='taxi environment')
    parser.add_argument('--nt', type=int, required=False, default=1)
    parser.add_argument('--ts', type=int, required=False, default=250000)
    parser.add_argument('--gm', type=float, required=False, default=1.0)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--save_file', required=True, type=str)
    args = parser.parse_args()
    args.val_writer = SummaryWriter(args.save_file)

    length = 5
    env = taxi(length)
    n_state = env.n_state
    n_action = env.n_action

    alpha = np.float(0.6)
    pi_eval = np.load('taxi-policy/pi19.npy')
    # shape: (2000, 6)
    pi_behavior = np.load('taxi-policy/pi3.npy')
    pi_behavior = alpha * pi_eval + (1 - alpha) * pi_behavior

    SASR_b_train = SASR_Dataset('matrix_b/train')
    SASR_b_val = SASR_Dataset('matrix_b/val')
    SASR_e = SASR_Dataset('matrix_e/test')

    train_loader = torch.utils.data.DataLoader(
        SASR_b_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        SASR_b_val, batch_size=args.batch_size, shuffle=True)

    eta = torch.FloatTensor(pi_eval / pi_behavior)

    w = SimpleCNN()
    f = SimpleCNN()
    for param in w.parameters():
        param.requires_grad = True
    for param in f.parameters():
        param.requires_grad = True
    if torch.cuda.is_available():
        w = w.cuda()
        f = f.cuda()
        eta = eta.cuda()

    # TODO: lower lr. lr depends on model
    # Want obj stablly, non-chaotic decreasing to 0. mean_mse is cheating
    w_optimizer = OAdam(w.parameters(), lr=1e-5, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=5e-5, betas=(0.5, 0.9))
    w_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(w_optimizer, step_size=800, gamma=0.1)
    f_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(f_optimizer, step_size=800, gamma=0.1)

    # SASR_b, b_freq, _ = roll_out(n_state, env, pi_behavior, 1, 10000000)
    # SASR_e, e_freq, _ = roll_out(n_state, env, pi_eval, 1, 10000000)
    # np.save('taxi7/taxi-policy/pi3_distrib.npy', b_freq/np.sum(b_freq))
    # np.save('taxi7/taxi-policy/pi19_distrib.npy', e_freq/np.sum(e_freq))

    b_freq_distrib = np.load('pi3_distrib.npy')
    e_freq_distrib = np.load('pi19_distrib.npy')

    gt_w = torch.Tensor(e_freq_distrib / b_freq_distrib)

    # estimate policy value using ground truth w
    s = torch.LongTensor([s_ for s_, _, _, _ in SASR_b_train.get_SASR()])
    a = torch.LongTensor([a_ for _, a_, _, _ in SASR_b_train.get_SASR()])
    r = torch.FloatTensor([r_ for _, _, _, r_ in SASR_b_train.get_SASR()])
    pi_b = torch.FloatTensor(pi_behavior)
    pi_e = torch.FloatTensor(pi_eval)
    gt_w_estimate = float((gt_w[s] * pi_e[s, a] / pi_b[s, a] * r).mean())
    print("estimate using ground truth w:", gt_w_estimate)
    del pi_b
    del pi_e

    # estimate policy value from evaluation policy roll out
    r_e = torch.FloatTensor([r_ for _, _, _, r_ in SASR_e.get_SASR()])
    roll_out_estimate = float(r_e.mean())
    print("estimate using evaluation policy roll-out:", roll_out_estimate)
    del r_e


    for epoch in range(5000):
        print(epoch) 
        dev_obj, dev_mse = validate(val_loader, w, f, args.gm, eta,
                                    roll_out_estimate, args, gt_w)
#         if epoch % args.print_freq == 0:
#             w_all = w(s_all).detach().flatten()
#             w_rmse = float(((w_all - gt_w) ** 2).mean() ** 0.5)
#             print("epoch %d, dev objective = %f, dev mse = %f, w rmse %f"
#                   % (epoch, dev_obj, dev_mse, w_rmse))
#             print("pred w:")
#             print(w_all[:20])
#             print("gt w:")
#             print(gt_w[:20])
#             print("mean w:", float(w(s).mean()))
            # torch.save({'w_model': w.state_dict(),
            #             'f_model': f.state_dict()},
            #            os.path.join(args.save_file, str(epoch) + '.pth'))
        train(train_loader, w, f, eta, w_optimizer, f_optimizer)
        w_optimizer_scheduler.step()
        f_optimizer_scheduler.step()
#         import pdb;pdb.set_trace()
        args.val_writer.add_scalar('lr',w_optimizer_scheduler.get_lr()[0], global_epoch) 
        global_epoch += 1


if __name__ == '__main__':
    main()
