from adversarial_learning.game_objectives import w_game_objective
from adversarial_learning.init_state_sampler import DiscreteInitStateSampler
from adversarial_learning.oadam import OAdam
from adversarial_learning.tau_list_dataset import TauListDataLoader
from debug_logging.w_logger import SimpleDiscretePrintWLogger
from environments.taxi_environment import TaxiEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.discrete_estimators import w_estimator_discrete
from models.discrete_models import StateEmbeddingModel
from models.w_adversary_wrapper import WAdversaryWrapper
from policies.discrete_policy import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy
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
    :param init_state_dist: tensor of shape (num_s,) containing probabilities
        of states at t=0
    :param val_tau_list: (optional) list of validation trajectories for logging
    :param val_freq: frequency of how often we perform validation logging (only
        if logger object is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractWLogger)
    :return: None
    """
    assert isinstance(f, WAdversaryWrapper)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)
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
    env = TaxiEnvironment()
    gamma = 0.98
    alpha = 0.6
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_other = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_other, pi_1_weight=alpha)

    # set up logger
    oracle_tau_len = 1000000
    init_state_dist_path = "taxi_data/init_state_dist.npy"
    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    logger = SimpleDiscretePrintWLogger(env=env, pi_e=pi_e, pi_b=pi_b,
                                        gamma=gamma,
                                        oracle_tau_len=oracle_tau_len)

    # generate train, val, and test data
    tau_len = 200000
    burn_in = 100000
    train_tau_list = env.generate_roll_out(pi=[pi_b], num_tau=1, tau_len=tau_len,
                                           burn_in=burn_in)
    val_tau_list = env.generate_roll_out(pi=[pi_b], num_tau=1, tau_len=tau_len,
                                         burn_in=burn_in)
    test_tau_list = env.generate_roll_out(pi=[pi_b], num_tau=1, tau_len=tau_len,
                                          burn_in=burn_in)

    # define networks and optimizers
    w = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    f = WAdversaryWrapper(StateEmbeddingModel(num_s=env.num_s, num_out=1))
    w_lr = 1e-3
    w_optimizer = OAdam(w.parameters(), lr=w_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=w_lr*5, betas=(0.5, 0.9))

    train_w_network(train_tau_list=train_tau_list, pi_e=pi_e, pi_b=pi_b,
                    num_epochs=1000, batch_size=1024, w=w, f=f,
                    w_optimizer=w_optimizer, f_optimizer=f_optimizer,
                    gamma=gamma, init_state_dist=init_state_dist,
                    val_tau_list=val_tau_list, val_freq=10, logger=logger)

    # calculate final performance
    policy_val_est = w_estimator_discrete(tau_list_data_loader=test_tau_list,
                                          pi_e=pi_e, pi_b=pi_b, w=w)
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    squared_error = (policy_val_est - policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()

