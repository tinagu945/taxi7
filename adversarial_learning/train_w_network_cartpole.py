import os
import torch
from torch.optim import Adam
from adversarial_learning.game_objectives import w_game_objective
from benchmark_methods.erm_w_benchmark import train_w_network_erm
from dataset.init_state_sampler import CartpoleInitStateSampler
from adversarial_learning.oadam import OAdam
from dataset.tau_list_dataset import TauListDataset, restore_dataset_from_load
from debug_logging.w_logger import TensorboardContinuousPrintWLogger
from environments.cartpole_environment import CartpoleEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import w_estimator_continuous
from models.continuous_models import QNetworkModel
from models.w_adversary_wrapper import WAdversaryWrapper
from policies.mixture_policies import GenericMixturePolicy
from policies.cartpole_policies import load_cartpole_policy
from train_w_network import train_w_network




def debug():
    # set up environment and policies
    env = CartpoleEnvironment()
    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50
    tau_e_path = 'tau_e_cartpole/'
    tau_b_path = 'tau_b_cartpole/'
    load= True
    
    pi_e = load_cartpole_policy("cartpole_weights/cartpole_best.pt", temp, env.state_dim,
                                hidden_dim, env.num_a)
    pi_other = load_cartpole_policy("cartpole_weights/cartpole_210_318.0.pt", temp,
                                    env.state_dim, hidden_dim, env.num_a)
    pi_b = GenericMixturePolicy(pi_e, pi_other, alpha)


    # set up logger
    oracle_tau_len = 1000000 #//100000
    # From gym repo, reset(): self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))  
    init_state_sampler = CartpoleInitStateSampler(env)
    logger = TensorboardContinuousPrintWLogger(env=env, pi_e=pi_e, pi_b=pi_b,
                                        gamma=gamma, hidden_dim=hidden_dim,
                                        oracle_tau_len=oracle_tau_len, load_path=tau_e_path)
    if load:
        train_data = restore_dataset_from_load(tau_b_path, '200000_train_')
        val_data = restore_dataset_from_load(tau_b_path, '200000_val_')
        test_data = restore_dataset_from_load(tau_b_path, '200000_test_')
        
    else:
        # generate train, val, and test data, very slow so load exisiting data is preferred.
        tau_len = 200000 #//100000
        burn_in = 100000 #//100000
        train_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
        val_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                         burn_in=burn_in)
        test_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                          burn_in=burn_in)
    
    # define networks and optimizers
    w = QNetworkModel(env.state_dim, hidden_dim)
    f = WAdversaryWrapper(QNetworkModel(env.state_dim, hidden_dim))
    w_lr_pre = 1e-4
    w_lr = 1e-6
    w_optimizer_pre = Adam(w.parameters(), lr=w_lr_pre, betas=(0.5, 0.9))
    w_optimizer = OAdam(w.parameters(), lr=w_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=w_lr*500, betas=(0.5, 0.9))

    train_w_network_erm(train_data=train_data, pi_e=pi_e, pi_b=pi_b,
                        num_epochs=50, batch_size=1024, w=w,
                        w_optimizer=w_optimizer_pre, gamma=gamma,
                        val_data=val_data, val_freq=10, logger=logger)
    train_w_network(train_data=train_data, pi_e=pi_e, pi_b=pi_b,
                    num_epochs=1000, batch_size=1024, w=w, f=f,
                    w_optimizer=w_optimizer, f_optimizer=f_optimizer,
                    gamma=gamma, init_state_sampler=init_state_sampler,
                    val_data=val_data, val_freq=10, logger=logger)

    # calculate final performance
    test_data_loader = test_data.get_data_loader(1024)
    policy_val_est = w_estimator_continuous(tau_list_data_loader=test_data_loader,
                                 pi_e=pi_e, pi_b=pi_b, w=w)
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000, load_path=tau_e_path)
    squared_error = (policy_val_est - policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()

