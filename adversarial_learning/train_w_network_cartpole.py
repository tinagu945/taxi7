import sys
sys.path.insert(0,'./')
import torch
from adversarial_learning.game_objectives import w_game_objective
from adversarial_learning.init_state_sampler import ContinuousInitStateSampler
from adversarial_learning.oadam import OAdam
from adversarial_learning.tau_list_dataset import TauListDataLoader
from debug_logging.w_logger import SimpleContinuousPrintWLogge
from environments.cartpole_environment import CartpoleEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.continuous_estimators import w_estimator_continuous
from models.continuous_models import QNetworkModel
from models.w_adversary_wrapper import WAdversaryWrapper
from policies.cartpole_policies import load_cartpole_policy
from utils.torch_utils import load_tensor_from_npy



def train_w_network(train_tau_list, pi_e, pi_b, num_epochs, batch_size, w, f,
                    w_optimizer, f_optimizer, gamma, init_state_dist,
                    val_tau_list=None, val_freq=10, logger=None):
    """
    :param train_tau_list: list of trajectories logged from behavior policy
        to use for training
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_b: behavior policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param w: initial w network (should be torch.nn.Module)
    :param f: initial f network (should be WAdversaryWrapper)
    :param w_optimizer: optimizer for w network
    :param f_optimizer: optimizer for f network
    :param gamma: discount factor (0 < gamma <= 1)
    :param init_state_dist: tensor of shape (2,) containing uniform sample range
    :param val_tau_list: (optional) list of validation trajectories for logging
    :param val_freq: frequency of how often we perform validation logging (only
        if logger object is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractWLogger)
    :return: None
    """
    assert isinstance(f, WAdversaryWrapper)
    init_state_sampler = ContinuousInitStateSampler(init_state_dist[0], init_state_dist[1])
    train_data_loader = TauListDataLoader(tau_list=train_tau_list,
                                          batch_size=batch_size)
    if val_tau_list:
        val_data_loader = TauListDataLoader(tau_list=val_tau_list,
                                            batch_size=batch_size)
    else:
        val_data_loader = None

    for epoch in range(num_epochs):
        for s, a, s_prime, r in train_data_loader:
            s_0 = init_state_sampler.get_sample(batch_size)
            w_obj, f_obj = w_game_objective(w=w, f=f, s=s, a=a, s_prime=s_prime,
                                            pi_e=pi_e, pi_b=pi_b, s_0=s_0,
                                            gamma=gamma)

            f_optimizer.zero_grad()
            f_obj.backward(retain_graph=True)
            f_optimizer.step()

            w_optimizer.zero_grad()
            w_obj.backward()
            w_optimizer.step()

        if logger and epoch % val_freq == 0:
            logger.log(train_data_loader, val_data_loader, w, f,
                       init_state_sampler, epoch)


def debug():
    # set up environment and policies
    env = CartpoleEnvironment()
    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 64
    pi_e = load_cartpole_policy("logs/cartpole_best.pt", temp)
    pi_other = load_cartpole_policy("logs/cartpole_210_318.0.pt", temp)
    pi_b = MixtureContinuousPolicy(pi_e, pi_other, alpha)
    
    

    # set up logger
    oracle_tau_len = 1000000
    # From gym repo, reset(): self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
    init_state_dist = [-0.05, 0.05]
    logger = SimpleContinuousPrintWLogger(env=env, pi_e=pi_e, pi_b=pi_b,
                                        gamma=gamma,
                                        oracle_tau_len=oracle_tau_len)

    # generate train, val, and test data
    tau_len = 200000
    burn_in = 100000
    train_tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
    val_tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                         burn_in=burn_in)
    test_tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                          burn_in=burn_in)

    # define networks and optimizers
    w = QNetworkModel(env.num_s, hidden_dim, env.num_a)
    f = WAdversaryWrapper(QNetworkModel(env.num_s, hidden_dim, env.num_a))
    w_lr = 1e-3
    w_optimizer = OAdam(w.parameters(), lr=w_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=w_lr*5, betas=(0.5, 0.9))

    train_w_network(train_tau_list=train_tau_list, pi_e=pi_e, pi_b=pi_b,
                    num_epochs=1000, batch_size=1024, w=w, f=f,
                    w_optimizer=w_optimizer, f_optimizer=f_optimizer,
                    gamma=gamma, init_state_dist=init_state_dist,
                    val_tau_list=val_tau_list, val_freq=10, logger=logger)

    # calculate final performance
    policy_val_est = w_estimator_continuous(tau_list_data_loader=test_tau_list,
                                          pi_e=pi_e, pi_b=pi_b, w=w)
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    squared_error = (policy_val_est - policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()

