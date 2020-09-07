import os
import datetime
import torch
from torch.optim import Adam
from benchmark_methods.erm_q_benchmark import train_q_network_erm
from dataset.init_state_sampler import CartpoleInitStateSampler
from adversarial_learning.oadam import OAdam
from dataset.tau_list_dataset import TauListDataset
from debug_logging.q_logger import ContinuousQLogger
from environments.cartpole_environment import CartpoleEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import q_estimator
from models.continuous_models import QNetworkModel
from policies.mixture_policies import GenericMixturePolicy
from policies.cartpole_policies import load_cartpole_policy
from train_q_network import train_q_network


def debug():
    # set up environment and policies
    env = CartpoleEnvironment()
    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50
    tau_e_path = 'tau_e_cartpole/'
    tau_b_path = 'tau_b_cartpole/'
    cartpole_weights = 'cartpole_weights/'
    load_train = False
    pi_other_name = 'cartpole_180_-99900.0'
    # Best reward -99589.0
    pi_e = load_cartpole_policy(os.path.join(cartpole_weights, "cartpole_best.pt"), temp, env.state_dim,
                                hidden_dim, env.num_a)
    pi_other = load_cartpole_policy(os.path.join(cartpole_weights, pi_other_name+".pt"), temp,
                                    env.state_dim, hidden_dim, env.num_a)  # cartpole_210_318.0.pt
    pi_b = GenericMixturePolicy(pi_e, pi_other, alpha)

    # set up logger
    oracle_tau_len = 100000  # // 100000
    # From gym repo, reset(): self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    init_state_sampler = CartpoleInitStateSampler(env)
    now = datetime.datetime.now()
    if load_train:
        print('Loading datasets for training...')
        train_data = TauListDataset.load(tau_b_path, pi_other_name+'_train_')
        val_data = TauListDataset.load(tau_b_path, pi_other_name+'_val_')
        test_data = TauListDataset.load(tau_b_path, pi_other_name+'_test_')
    else:
        # generate train, val, and test data, very slow so load exisiting data is preferred.
        tau_len = 200000  # // 100000
        burn_in = 100000  # // 100000
        print('not loading train b data, so generating')
        train_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
        print('train done'+pi_other_name+'_train_')
        val_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                         burn_in=burn_in)
        print('val done')
        test_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                          burn_in=burn_in)
        print('test done')
        # train_data.save(tau_b_path, prefix=pi_other_name+'_train_')
        # val_data.save(tau_b_path, prefix=pi_other_name+'_val_')
        # test_data.save(tau_b_path, prefix=pi_other_name+'_test_')
        # print('all saved!', pi_other_name)

        # print('not loading train e data, so generating')
        # train_data_e = env.generate_roll_out(
        #     pi=pi_e, num_tau=1, tau_len=tau_len, burn_in=burn_in)
        # print('train done for e')
        # val_data_e = env.generate_roll_out(pi=pi_e, num_tau=1, tau_len=tau_len,
        #                                    burn_in=burn_in)
        # print('val done for e')
        # test_data_e = env.generate_roll_out(pi=pi_e, num_tau=1, tau_len=tau_len,
        #                                     burn_in=burn_in)
        # print('test done for e')
        # train_data_e.save(tau_e_path)
        # val_data_e.save(tau_e_path)
        # test_data_e.save(tau_e_path)
        # print('all saved for pi_e!')

    # print('Train pi_b policy value', on_policy_estimate(
    #     env=env, pi_e=pi_b, gamma=gamma, num_tau=1, tau_len=1000000))

    logger = ContinuousQLogger(env=env, pi_e=pi_e, pi_b=pi_b, gamma=gamma, tensorboard=True, oracle_tau_len=oracle_tau_len, oracle_path=os.path.join(
        cartpole_weights, "cartpole_best.pt"), load_path=tau_e_path, save_model=True, init_state_sampler=init_state_sampler)
    # define networks and optimizers
    q = QNetworkModel(env.state_dim, hidden_dim, out_dim=env.num_a)
    f = QNetworkModel(env.state_dim, hidden_dim, out_dim=env.num_a)
    q_lr_pre = 1e-2
    q_lr = 1e-7
    q_optimizer_pre = Adam(q.parameters(), lr=q_lr_pre, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=q_lr*500, betas=(0.5, 0.9))
    # train_q_network_erm(train_data=train_data, pi_e=pi_e,
    #                     num_epochs=1000, batch_size=1024, q=q,
    #                     q_optimizer=q_optimizer_pre, gamma=gamma,
    #                     val_data=val_data, val_freq=10, logger=logger)

    q_new = QNetworkModel.load_continuous_q(env, hidden_dim, env.num_a, os.path.join(
        'logs/2020-09-04T07:12:06.822533', 'best_q_ERM.pt'), cuda=False)  # logger.path
    q.model.train()
    q_optimizer = OAdam(q_new.parameters(), lr=q_lr, betas=(0.5, 0.9))
    print('Finished ERM pretraining, now loading its best model.')
    train_q_network(train_data=train_data, pi_e=pi_e,
                    num_epochs=2000, batch_size=1024, q=q_new, f=f,
                    q_optimizer=q_optimizer, f_optimizer=f_optimizer,
                    gamma=gamma,
                    val_data=val_data, val_freq=10, logger=logger)

    # calculate final performance, optional
    policy_val_est = q_estimator(
        pi_e=pi_e, gamma=gamma, q=q_new, init_state_sampler=init_state_sampler)
    # policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
    #                                        num_tau=1, tau_len=1000000, load_path=tau_e_path)
    squared_error = (policy_val_est - logger.policy_val_oracle) ** 2
    print('Policy_val_oracle', logger.policy_val_oracle)
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()
