import torch
import datetime
from dataset.init_state_sampler import CartpoleInitStateSampler
from dataset.tau_list_dataset import TauListDataset
from environments.cartpole_environment import CartpoleEnvironment
from estimators.benchmark_estimators import *
from estimators.infinite_horizon_estimators import *
from benchmark_methods.erm_q_benchmark import *
from models.continuous_models import *
from models.w_adversary_wrapper import WAdversaryWrapper
from policies.mixture_policies import GenericMixturePolicy
from policies.cartpole_policies import load_cartpole_policy
from debug_logging.q_logger import ContinuousQLogger, SimplestQLogger
from debug_logging.w_logger import ContinuousWLogger, SimplestWLogger
from all_compare import Density_Ratio_discounted
from benchmark_methods.continuous_w_benchmark import RKHS_method, gaussian_kernel
from adversarial_learning.train_q_network_cartpole import train_q_network_cartpole
from adversarial_learning.train_w_network_cartpole import train_w_network_cartpole
from adversarial_learning.oadam import OAdam
import argparse
import sys
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument('--c_reg', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--temp', type=float, default=2.0)
parser.add_argument('--oracle_val', type=float, default=-179.2)
parser.add_argument('--hidden_dim', type=int, default=50)
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--reward_reshape', action='store_true', default=False)
parser.add_argument('--pi_other_name', type=str,
                    default='400_-99761_cartpole')
parser.add_argument('--pi_e_name', type=str,
                    default='456_-99501_cartpole_best')
parser.add_argument('--tau_e_path', type=str, default='tau_e_cartpole')
parser.add_argument('--tau_b_path', type=str, default='tau_b_cartpole')
parser.add_argument('--cartpole_weights', type=str,
                    default='cartpole_weights_1')
parser.add_argument('--q_ERM_epoch', type=int, default=0)
parser.add_argument('--q_GMM_epoch', type=int, default=150)
parser.add_argument('--w_ERM_epoch', type=int, default=150)
parser.add_argument('--w_GMM_epoch', type=int, default=100)
parser.add_argument('--w_RKHS_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--q_pre_lr', type=float, default=1e-2)
parser.add_argument('--q_lr', type=float, default=1e-3)
parser.add_argument('--f_lr_multiple', type=float, default=1.0)
parser.add_argument('--w_pre_lr', type=float, default=1e-3)
parser.add_argument('--w_lr', type=float, default=1e-4)
parser.add_argument('--w_rkhs_lr', type=float, default=5e-3)
parser.add_argument('--val_freq', type=int, default=10)
parser.add_argument('--save_folder', type=str, default='logs')

args = parser.parse_args()
args.script = 'Cartpole_complete'
print(args)
end = 15


def debug():
    env = CartpoleEnvironment(reward_reshape=args.reward_reshape)
    # [50000, 100000, 200000, 400000] // 10000
    tau_lens = [400000]
    burn_in = 100000  # // 10000

    pi_e = load_cartpole_policy(os.path.join(args.cartpole_weights, args.pi_e_name+".pt"), args.temp, env.state_dim,
                                args.hidden_dim, env.num_a)
    pi_other = load_cartpole_policy(os.path.join(args.cartpole_weights, args.pi_other_name+".pt"), args.temp,
                                    env.state_dim, args.hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, args.alpha)
    init_state_sampler = CartpoleInitStateSampler(env)
    print('policy_val_oracle', args.oracle_val)

    # combinations=['is_est', 'is_ess', 'q_ERM', 'w_RKHS', 'drl_q_ERM_w_RKHS', 'q_GMM', 'w_GMM', 'drl_q_GMM_w_RKHS', 'drl_q_ERM_w_GMM', 'drl_q_GMM_w_GMM']

    for j in tau_lens:
        tau_len = j
        preds = []
        for i in range(10, end):
            print(j, i)
            np.random.seed(i)
            torch.random.manual_seed(i)
            random.seed(i)

            train_data = env.generate_roll_out(
                pi=pi_b, num_tau=1, tau_len=tau_len, burn_in=args.burn_in)
            # train_data = TauListDataset.load(
            #     args.tau_b_path, prefix=args.pi_other_name+'_train_')
            train_data_loader = train_data.get_data_loader(args.batch_size)
            SASR = train_data.restore_strcture(discrete=False)
            print('finished data generation.')
            # policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma, num_tau=1, tau_len=10000000)

            is_est = importance_sampling_estimator(
                SASR, pi_b, pi_e, args.gamma, split_shape=[4, 1, 4, 1])
            is_ess = importance_sampling_estimator_stepwise(
                SASR, pi_b, pi_e, args.gamma, split_shape=[4, 1, 4, 1])
            print('is_est', is_est, 'is_ess', is_ess)

            # RKHS for w
            now = datetime.datetime.now()
            timestamp = now.isoformat()
            log_path = '{}/{}/'.format(args.save_folder,
                                       '_'.join([timestamp]+[i.replace("--", "") for i in sys.argv[1:]]+[args.script+'_W_'+str(j)+str(i)]))
            w_logger = SimplestWLogger(env, pi_e, pi_b, args.gamma,
                                       True, True, log_path, args.oracle_val)
            # w_logger = None

            w_rkhs = RKHS_method(env, pi_b, pi_e, train_data, args.gamma,
                                 args.hidden_dim, init_state_sampler, gaussian_kernel, logger=w_logger)  # , batch_size=args.batch_size, epoch=args.w_RKHS_epoch, lr=args.w_rkhs_lr)

            w_rkhs_est = w_estimator(train_data_loader, pi_e, pi_b, w_rkhs)
            print('w_rkhs_est', w_rkhs_est)

            # ERM
            now = datetime.datetime.now()
            timestamp = now.isoformat()
            log_path = '{}/{}/'.format(args.save_folder,
                                       '_'.join([timestamp]+[i.replace("--", "") for i in sys.argv[1:]]+[args.script+'_Q_'+str(j)+str(i)]))
            q_logger = SimplestQLogger(
                env, pi_e, args.gamma, init_state_sampler, True, True, log_path, args.oracle_val)
            # q_logger = None

            q_erm = QNetworkModelSimple(
                env.state_dim, args.hidden_dim, out_dim=env.num_a, neg_output=True)
            q_erm_optimizer = OAdam(
                q_erm.parameters(), lr=args.q_lr, betas=(0.5, 0.9))
            train_q_network_erm(train_data, pi_e, args.q_GMM_epoch+args.q_ERM_epoch,
                                args.batch_size, q_erm, q_erm_optimizer, args.gamma,
                                val_data=None, val_freq=10,
                                q_scheduler=None, logger=q_logger)
            q_erm_est = q_estimator(
                pi_e, args.gamma, q_erm, init_state_sampler)
            print('q_erm_est', q_erm_est)

            # q GMM
            q_GMM = train_q_network_cartpole(
                args, env, train_data, None, pi_e, pi_b, init_state_sampler, q_logger)
            q_GMM_est = q_estimator(
                pi_e, args.gamma, q_GMM, init_state_sampler)
            print('q_GMM_est', q_GMM_est)

            # w GMM
            w_GMM = train_w_network_cartpole(
                args, env, train_data, None, pi_e, pi_b, init_state_sampler, w_logger)
            w_GMM_est = w_estimator(train_data_loader, pi_e, pi_b, w_GMM)
            print('w_GMM_est', w_GMM_est)

            drl_q_ERM_w_RKHS = double_estimator(
                train_data_loader, pi_e, pi_b, w_rkhs, q_erm, args.gamma, init_state_sampler)
            drl_q_ERM_w_GMM = double_estimator(
                train_data_loader, pi_e, pi_b, w_GMM, q_erm, args.gamma, init_state_sampler)
            drl_q_GMM_w_RKHS = double_estimator(
                train_data_loader, pi_e, pi_b, w_rkhs, q_GMM, args.gamma, init_state_sampler)
            drl_q_GMM_w_GMM = double_estimator(
                train_data_loader, pi_e, pi_b, w_GMM, q_GMM, args.gamma, init_state_sampler)
            print('drl_q_ERM_w_RKHS', drl_q_ERM_w_RKHS, 'drl_q_ERM_w_GMM', drl_q_ERM_w_GMM,
                  'drl_q_GMM_w_RKHS', drl_q_GMM_w_RKHS, 'drl_q_GMM_w_GMM', drl_q_GMM_w_GMM)

            preds.append([is_est, is_ess, w_rkhs_est, q_erm_est, q_GMM_est, w_GMM_est,
                          drl_q_ERM_w_RKHS, drl_q_ERM_w_GMM, drl_q_GMM_w_RKHS, drl_q_GMM_w_GMM])

        print('[is_est, is_ess, w_rkhs_est, q_erm_est, q_GMM_est, w_GMM_est, drl_q_ERM_w_RKHS, drl_q_ERM_w_GMM, drl_q_GMM_w_RKHS, drl_q_GMM_w_GMM] \n', preds)
        preds = np.array(preds)
        errors = (preds-args.oracle_val)**2
        mse = np.mean(errors, axis=0)
        print('MSE for [is_est, is_ess, w_rkhs_est, q_erm_est, q_GMM_est, w_GMM_est, drl_q_ERM_w_RKHS, drl_q_ERM_w_GMM, drl_q_GMM_w_RKHS, drl_q_GMM_w_GMM] \n', mse)
        np.save('estimators/benchmarks_GMM_cont_'+str(j)+str(end), preds)
        np.save('estimators/benchmarks_GMM_cont_'+str(j)+str(end), mse)


if __name__ == "__main__":
    debug()
