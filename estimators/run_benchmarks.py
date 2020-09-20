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
from estimators.benchmark_estimators import *
from estimators.infinite_horizon_estimators import *
import random
import numpy as np


def debug():
    env = TaxiEnvironment(discrete_state=True)
    gamma = 0.98
    alpha = 0.6
    tau_lens = [50000, 100000, 200000, 400000]  # // 10000
    burn_in = 0  # 100000  # // 10000
    init_state_dist_path = "taxi_data/init_state_dist.npy"

    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_s = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)

    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)
    policy_val_oracle = -0.74118131399
    print('policy_val_oracle', policy_val_oracle)

    for j in tau_lens:
        tau_len = j
        preds = []
        for i in range(100):
            print(j, i)
            np.random.seed(i)
            torch.random.manual_seed(i)
            random.seed(i)

            train_data = env.generate_roll_out(
                pi=pi_b, num_tau=1, tau_len=tau_len, burn_in=burn_in)
            train_data_loader = train_data.get_data_loader(1024)
            SASR = train_data.restore_strcture(discrete=True)
            # print('finished data generation.')
            # policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
            #                                        num_tau=1, tau_len=10000000)

            is_est = importance_sampling_estimator(SASR, pi_b, pi_e, gamma)
            is_ess = importance_sampling_estimator_stepwise(
                SASR, pi_b, pi_e, gamma)
            # print('is_est', is_est, 'is_ess', is_ess)

            # # Masa's w method, only for discrete
            den_discrete = Density_Ratio_discounted(env.n_state, gamma)
            _, w_table = train_density_ratio(
                SASR, pi_b, pi_e, den_discrete, gamma)
            w_table = w_table.reshape(-1)
            w = StateEmbeddingModel(num_s=env.num_s, num_out=1)
            w.set_weights(torch.from_numpy(w_table))
            # print('[Masa] fitted w')

            # # Masa's q method is the same as fit_q_tabular
            q = fit_q_tabular(train_data, pi_e, gamma)
            # print('[Masa] fitted q')

            q_est = q_estimator(pi_e, gamma, q, init_state_sampler)
            drl_est = double_estimator(
                train_data_loader, pi_e, pi_b, w, q, gamma, init_state_sampler)
            w_est = w_estimator(train_data_loader, pi_e, pi_b, w)
            # print('[Masa] q_est', q_est, '[Masa] w_est',
            #   w_est, '[Masa] drl_est', drl_est)

            preds.append([is_est, is_ess, q_est, drl_est, w_est])

        preds = np.array(preds)
        errors = (preds-policy_val_oracle)**2
        mse = np.mean(errors, axis=0)
        print('[is_est, is_ess, q_est, drl_est, w_est] \n', preds)
        print('MSE for [is_est, is_ess, q_est, drl_est, w_est] \n', mse)
        np.save('estimators/masa_preds_'+str(j), preds)
        np.save('estimators/masa_mse_'+str(j), mse)


if __name__ == "__main__":
    debug()
