import os
import datetime
import torch
from torch.optim import Adam
from benchmark_methods.erm_w_benchmark import train_w_network_erm
from dataset.init_state_sampler import CartpoleInitStateSampler
from adversarial_learning.oadam import OAdam
from dataset.tau_list_dataset import TauListDataset
from debug_logging.w_logger import ContinuousWLogger, SimplestWLogger
from environments.cartpole_environment import CartpoleEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import w_estimator
from models.continuous_models import WNetworkModel
from models.w_adversary_wrapper import WAdversaryWrapper
from policies.mixture_policies import GenericMixturePolicy
from policies.cartpole_policies import load_cartpole_policy
from train_w_network import train_w_network
from torch.optim import lr_scheduler
import argparse
import datetime
import sys
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--temp', type=float, default=2.0)
parser.add_argument('--w_pre_lr', type=float, default=1e-3)
parser.add_argument('--w_lr', type=float, default=1e-4)
parser.add_argument('--f_lr_multiple', type=float, default=1)
parser.add_argument('--hidden_dim', type=int, default=50)
parser.add_argument('--lr_decay', type=int, default=np.inf)
parser.add_argument('--num_tau', type=int, default=1)
parser.add_argument('--oracle_tau_len', type=int, default=1000000)
parser.add_argument('--tau_len', type=int, default=200000)
parser.add_argument('--burn_in', type=int, default=100000)
parser.add_argument('--ERM_epoch', type=int, default=200)
parser.add_argument('--GMM_epoch', type=int, default=500)
parser.add_argument('--val_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--load_train', action='store_false', default=True)
parser.add_argument('--no_tensorboard', action='store_false', default=True)
parser.add_argument('--no_save_model', action='store_false', default=True)
parser.add_argument('--cuda', action='store_true', default=False)
# Best reward -99589.0
parser.add_argument('--pi_other_name', type=str,
                    default='cartpole_180_-99900.0')
parser.add_argument('--tau_e_path', type=str, default='tau_e_cartpole')
parser.add_argument('--tau_b_path', type=str, default='tau_b_cartpole')
parser.add_argument('--cartpole_weights', type=str, default='cartpole_weights')
parser.add_argument('--save_folder', type=str, default='logs')
args = parser.parse_args()
args.script = 'Wcartpole'


def debug():
    env = CartpoleEnvironment()
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    log_path = '{}/{}/'.format(args.save_folder,
                               '_'.join([timestamp]+[i.replace("--", "") for i in sys.argv[1:]]+[args.script]))

    # set up environment and policies
    pi_e = load_cartpole_policy(os.path.join(args.cartpole_weights, "cartpole_best.pt"), args.temp, env.state_dim,
                                args.hidden_dim, env.num_a)
    pi_other = load_cartpole_policy(os.path.join(args.cartpole_weights, args.pi_other_name+".pt"), args.temp,
                                    env.state_dim, args.hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, args.alpha)
    init_state_sampler = CartpoleInitStateSampler(env)
    now = datetime.datetime.now()
    if args.load_train:
        print('Loading datasets for training...')
        train_data = TauListDataset.load(
            args.tau_b_path, prefix=args.pi_other_name+'_train_')
        val_data = TauListDataset.load(
            args.tau_b_path, prefix=args.pi_other_name+'_val_')
        test_data = TauListDataset.load(
            args.tau_b_path, prefix=args.pi_other_name+'_test_')
        pi_e_data_discounted = TauListDataset.load(args.tau_e_path,
                                                   prefix="gamma_")
    else:
        # generate train, val, and test data, very slow so load exisiting data is preferred.
        # pi_b data is for training so no gamma.
        print('Not loading pi_b data, so generating')
        train_data = env.generate_roll_out(pi=pi_b, num_tau=args.num_tau, tau_len=args.tau_len,
                                           burn_in=args.burn_in)
        print('Finished generating train data of', args.pi_other_name)
        val_data = env.generate_roll_out(pi=pi_b, num_tau=args.num_tau, tau_len=args.tau_len,
                                         burn_in=args.burn_in)
        print('Finished generating val data of', args.pi_other_name)
        test_data = env.generate_roll_out(pi=pi_b, num_tau=args.num_tau, tau_len=args.tau_len,
                                          burn_in=args.burn_in)
        print('Finished generating test data of', args.pi_other_name)
        train_data.save(args.tau_b_path, prefix=args.pi_other_name+'_train_')
        val_data.save(args.tau_b_path, prefix=args.pi_other_name+'_val_')
        test_data.save(args.tau_b_path, prefix=args.pi_other_name+'_test_')

        # pi_e data with gamma is only for calculating oracle policy value, so has the gamma.
        print('Not loarding pi_e data, so generating')
        pi_e_data_discounted = env.generate_roll_out(
            pi=pi_e, num_tau=args.num_tau, tau_len=args.oracle_tau_len, burn_in=args.burn_in, gamma=args.gamma)
        print('Finished generating data of pi_e with gamma')
        pi_e_data_discounted.save(args.tau_e_path, prefix='gamma_')

    logger = SimplestWLogger(env, pi_e, pi_b, args.gamma,
                             True, True, log_path, -40.1)
    with open(os.path.join(log_path, 'meta.txt'), 'w') as f:
        print(args, file=f)

    # define networks and optimizers
    w = WNetworkModel(env.state_dim, args.hidden_dim, positive_output=True)
    f = WAdversaryWrapper(WNetworkModel(env.state_dim, args.hidden_dim))
    w_optimizer_pre = Adam(w.parameters(), lr=args.w_pre_lr)
    w_optimizer = OAdam(w.parameters(), lr=args.w_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=args.w_lr *
                        args.f_lr_multiple, betas=(0.5, 0.9))

    w_scheduler = lr_scheduler.StepLR(w_optimizer, step_size=args.lr_decay)
    f_scheduler = lr_scheduler.StepLR(f_optimizer, step_size=args.lr_decay)

    train_w_network_erm(train_data=train_data, pi_e=pi_e, pi_b=pi_b,
                        num_epochs=args.ERM_epoch, batch_size=args.batch_size, w=w,
                        w_optimizer=w_optimizer_pre, gamma=args.gamma,
                        val_data=val_data, val_freq=args.val_freq, logger=logger)
    print('Finished ERM pretraining.')
    train_w_network(train_data=train_data, pi_e=pi_e, pi_b=pi_b,
                    num_epochs=args.GMM_epoch, batch_size=args.batch_size, w=w, f=f,
                    w_optimizer=w_optimizer, f_optimizer=f_optimizer,
                    gamma=args.gamma, init_state_sampler=init_state_sampler,
                    val_data=val_data, val_freq=args.val_freq, logger=logger,
                    w_scheduler=w_scheduler, f_scheduler=f_scheduler)

    # calculate final performance
    policy_val_est = w_estimator(tau_list_data_loader=train_data_loader,
                                 pi_e=pi_e, pi_b=pi_b, w=w)
    squared_error = (policy_val_est - logger.policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()
