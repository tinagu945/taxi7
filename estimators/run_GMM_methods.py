import torch
from environments.taxi_environment import TaxiEnvironment
from policies.mixture_policies import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy
from all_compare import Density_Ratio_discounted
from benchmark_methods.discrete_q_benchmark import fit_q_tabular
from dataset.init_state_sampler import DiscreteInitStateSampler
from utils.torch_utils import load_tensor_from_npy
from models.discrete_models import StateEmbeddingModel, QTableModel
from models.w_adversary_wrapper import WAdversaryWrapper
# from estimators.benchmark_estimators import *
from estimators.infinite_horizon_estimators import double_estimator, w_estimator, q_estimator
import random
import numpy as np
import argparse
import os
import datetime
from debug_logging.q_logger import SimplestQLogger
from debug_logging.w_logger import SimplestWLogger
from adversarial_learning.train_q_network import train_q_taxi
from adversarial_learning.train_w_network import train_w_taxi


parser = argparse.ArgumentParser()
parser.add_argument('--tau_len', type=int, default=50000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--q_ERM_epoch', type=int, default=100)
parser.add_argument('--q_epoch', type=int, default=800)
parser.add_argument('--w_ERM_epoch', type=int, default=150)
parser.add_argument('--w_epoch', type=int, default=900)
parser.add_argument('--q', action='store_true', default=False)
parser.add_argument('--w', action='store_true', default=False)
args = parser.parse_args()


def debug():
    env = TaxiEnvironment(discrete_state=True)
    gamma = 0.98
    alpha = 0.6
    # tau_lens = [50000, 100000, 200000, 400000]  # // 10000
    tau_len = args.tau_len
    burn_in = 0  # 100000  # // 10000

    init_state_dist_path = "taxi_data/init_state_dist.npy"
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_s = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)

    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)
    policy_val_oracle = -0.74118131399
    print('policy_val_oracle', policy_val_oracle)

    # for j in tau_lens:
    # tau_len = j
    # preds = []
    # for i in range(100):
    #     print(j, i)
    for i in range(3):
        args.seed = i
        print('args.seed', args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        random.seed(args.seed)

        train_data = env.generate_roll_out(
            pi=pi_b, num_tau=1, tau_len=tau_len, burn_in=burn_in)
        train_data_loader = train_data.get_data_loader(1024)
        val_data = None
        print('train_data generated')

        now = datetime.datetime.now()
        if args.q:
            q_path = os.path.join('logs', '_'.join(
                [str(now.isoformat()), 'q', str(args.seed), str(args.tau_len)]))
            q_logger = SimplestQLogger(
                env, pi_e, gamma, init_state_sampler, True, True, q_path, policy_val_oracle)
            q = train_q_taxi(env, train_data, val_data, pi_e, pi_b,
                             init_state_sampler, q_logger, gamma, ERM_epoch=args.q_ERM_epoch, epoch=args.q_epoch)
            q_est = q_estimator(pi_e, gamma, q, init_state_sampler)
            print('[ours] q_est', q_est)
            with open(os.path.join(q_path, 'results.txt'), 'w') as f:
                f.write(str(q_est))
            print('q_est written in', q_path)

        if args.w:
            w_path = os.path.join('logs', '_'.join(
                [str(now.isoformat()), 'w', str(args.seed), str(args.tau_len)]))
            w_logger = SimplestWLogger(
                env, pi_e, pi_b, gamma, True, True, w_path, policy_val_oracle)
            w = train_w_taxi(env, train_data, val_data, pi_e, pi_b,
                             init_state_sampler, w_logger, gamma, ERM_epoch=args.w_ERM_epoch, epoch=args.w_epoch)
            w_est = w_estimator(train_data_loader, pi_e, pi_b, w)
            print('[ours] w_est', w_est)
            with open(os.path.join(w_path, 'results.txt'), 'w') as f:
                f.write(str(w_est))
            print('w_est written in', w_path)

        if args.w and args.q:
            double_est = double_estimator(
                train_data_loader, pi_e, pi_b, w, q, gamma, init_state_sampler)
            print('[ours] drl_est', double_est)
            with open(os.path.join(q_path, 'results.txt'), 'a') as f:
                f.write(str(double_est))
            print('double_est written in', q_path)


def load_eval():
    env = TaxiEnvironment(discrete_state=True)
    gamma = 0.98
    alpha = 0.6
    # tau_lens = [50000, 100000, 200000, 400000]  # // 10000
    tau_len = 200000  # args.tau_len
    burn_in = 0  # 100000  # // 10000

    init_state_dist_path = "taxi_data/init_state_dist.npy"
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_s = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)

    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)
    policy_val_oracle = -0.74118131399
    print('policy_val_oracle', policy_val_oracle)

    print('args.seed', args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)

    train_data = env.generate_roll_out(
        pi=pi_b, num_tau=1, tau_len=tau_len, burn_in=burn_in)
    train_data_loader = train_data.get_data_loader(1024)
    print('train_data generated')

    w_path = 'logs/2020-09-13T18:33:35.909858_w_2_200000/290_w_.pt'
    q_path = 'logs/2020-09-11T11:04:53.345691_q_2_200000/790_q_.pt'
    q_table = torch.zeros((env.num_s, env.num_a))
    q = QTableModel(q_table)
    q.model.load_state_dict(torch.load(q_path))
    w = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    w.load_state_dict(torch.load(w_path))
    print('[ours] w_oracle and q loaded!')
    double_est = double_estimator(
        train_data_loader, pi_e, pi_b, w, q, gamma, init_state_sampler)
    print('[ours] drl_est', double_est)


if __name__ == "__main__":
    # debug()
    load_eval()
