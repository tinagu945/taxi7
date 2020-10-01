import os
import datetime
import torch
from torch.optim import Adam
from benchmark_methods.erm_q_benchmark import train_q_network_erm
from dataset.init_state_sampler import CartpoleInitStateSampler
from adversarial_learning.oadam import OAdam
from dataset.tau_list_dataset import TauListDataset
from debug_logging.q_logger import ContinuousQLogger, SimplestQLogger
from environments.cartpole_environment import CartpoleEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import q_estimator
from models.continuous_models import QNetworkModel, QOracleModel, \
    QNetworkModelSimple
from policies.mixture_policies import GenericMixturePolicy
from policies.cartpole_policies import load_cartpole_policy
from train_q_network import train_q_network
from torch.optim import lr_scheduler
import argparse
import datetime
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--temp', type=float, default=2.0)
parser.add_argument('--q_pre_lr', type=float, default=1e-2)
parser.add_argument('--q_lr', type=float, default=1e-3)
parser.add_argument('--f_lr_multiple', type=float, default=1.0)
parser.add_argument('--hidden_dim', type=int, default=50)
parser.add_argument('--lr_decay', type=int, default=np.inf)
parser.add_argument('--num_tau', type=int, default=1)
parser.add_argument('--oracle_tau_len', type=int, default=1000000)
parser.add_argument('--tau_len', type=int, default=200000)
parser.add_argument('--burn_in', type=int, default=100000)
parser.add_argument('--ERM_epoch', type=int, default=100)
parser.add_argument('--GMM_epoch', type=int, default=2000)
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
args.script = 'Qcartpole'


def debug():
    env = CartpoleEnvironment()
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    log_path = '{}/{}/'.format(args.save_folder,
                               '_'.join([timestamp]+[i.replace("--", "") for i in sys.argv[1:]]+[args.script]))
    print(log_path)

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
        train_data.save(args.tau_b_path, prefix=args.pi_other_name+'_train_')
        val_data.save(args.tau_b_path, prefix=args.pi_other_name+'_val_')

        # pi_e data with gamma is only for calculating oracle policy value, so has the gamma.
        print('Not loarding pi_e data, so generating')
        pi_e_data_discounted = env.generate_roll_out(
            pi=pi_e, num_tau=args.num_tau, tau_len=args.oracle_tau_len, burn_in=args.burn_in, gamma=args.gamma)
        print('Finished generating data of pi_e with gamma')
        pi_e_data_discounted.save(args.tau_e_path, prefix='gamma_')

    q_oracle = QOracleModel.load_continuous_q_oracle(
        env, args.hidden_dim, env.num_a, 'logs_b4_0920/2020-09-11T13:38:31.250247_q_pre_lr_1e-4/best_q_ERM.pt')
    logger = ContinuousQLogger(env=env, pi_e=pi_e, gamma=args.gamma, tensorboard=args.no_tensorboard, save_model=args.no_save_model,
                               init_state_sampler=init_state_sampler, log_path=log_path, policy_val_oracle=-40.1, q_oracle=q_oracle, pi_e_data_discounted=pi_e_data_discounted)
    # logger = SimplestQLogger(
    #     env, pi_e, args.gamma, init_state_sampler, True, True, log_path, -40.1)

    with open(os.path.join(log_path, 'meta.txt'), 'w') as f:
        print(args, file=f)

    q = QNetworkModelSimple(env.state_dim, args.hidden_dim, out_dim=env.num_a,
                            neg_output=True)
    f = QNetworkModelSimple(env.state_dim, args.hidden_dim, out_dim=env.num_a)
    q_pre_optimizer = Adam(q.parameters(), lr=args.q_pre_lr)
    q_optimizer = OAdam(q.parameters(), lr=args.q_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=args.q_lr * args.f_lr_multiple,
                        betas=(0.5, 0.9))

    # q_pre_scheduler = lr_scheduler.StepLR(q_pre_optimizer, step_size=args.lr_decay)
    q_pre_scheduler = None
    # q_scheduler = lr_scheduler.StepLR(q_optimizer, step_size=args.lr_decay)
    # f_scheduler = lr_scheduler.StepLR(f_optimizer, step_size=args.lr_decay)
    q_scheduler = None
    f_scheduler = None
    mean_reward = float(train_data.r.mean())
    err_reward = float(train_data.r.std()) / (len(train_data.r) ** 0.5) * 1.96
    print("mean train reward: %f +- %f" % (mean_reward, err_reward))

    train_q_network_erm(train_data=train_data, pi_e=pi_e,
                        num_epochs=args.ERM_epoch, batch_size=args.batch_size, q=q,
                        q_optimizer=q_pre_optimizer, gamma=args.gamma,
                        val_data=val_data, val_freq=args.val_freq, logger=logger,
                        q_scheduler=q_pre_scheduler)  # train_data

    print('Finished ERM pretraining.')
    train_q_network(train_data=train_data, pi_e=pi_e,
                    num_epochs=args.GMM_epoch, batch_size=args.batch_size, q=q, f=f,
                    q_optimizer=q_optimizer, f_optimizer=f_optimizer,
                    gamma=args.gamma,
                    val_data=val_data, val_freq=args.val_freq, logger=logger,
                    q_scheduler=q_scheduler, f_scheduler=f_scheduler)

    # calculate final performance, optional
    policy_val_est = q_estimator(
        pi_e=pi_e, gamma=args.gamma, q=q, init_state_sampler=init_state_sampler)
    squared_error = (policy_val_est - logger.policy_val_oracle) ** 2
    print('Policy_val_oracle', logger.policy_val_oracle)
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()
