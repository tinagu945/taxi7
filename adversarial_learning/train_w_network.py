from torch.optim import Adam
from adversarial_learning.game_objectives import w_game_objective
from benchmark_methods.erm_w_benchmark import train_w_network_erm
from dataset.init_state_sampler import DiscreteInitStateSampler, \
    DecodingDiscreteInitStateSampler, CartpoleInitStateSampler
from adversarial_learning.oadam import OAdam
from dataset.tau_list_dataset import TauListDataset
from debug_logging.w_logger import DiscreteWLogger
from environments.cartpole_environment import CartpoleEnvironment
from environments.taxi_environment import TaxiEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import w_estimator
from models.cnn_models import TaxiSimpleCNN
from models.discrete_models import StateEmbeddingModel
from models.w_adversary_wrapper import WAdversaryWrapper
from policies.cartpole_policies import load_cartpole_policy
from policies.mixture_policies import GenericMixturePolicy, \
    MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy_continuous, load_taxi_policy
from utils.torch_utils import load_tensor_from_npy


def train_w_network(train_data, pi_e, pi_b, num_epochs, batch_size, w, f,
                    w_optimizer, f_optimizer, gamma, init_state_sampler,
                    val_data=None, val_freq=10, logger=None, w_scheduler=None, f_scheduler=None):
    """
    :param train_data: dataset logged from behavior policy used for training
        (should be instance of TauListDataset)
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_b: behavior policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param w: initial w network (should be torch.nn.Module)
    :param f: initial f network (should be WAdversaryWrapper)
    :param w_optimizer: optimizer for w network
    :param f_optimizer: optimizer for f network
    :param gamma: discount factor (0 < gamma <= 1)
    :param init_state_sampler: should subclass AbstractInitStateSampler,
        used for sampling states at t=0 or computing/estimating expectations
        w.r.t. this distribution
    :param val_data: (optional) additional dataset logged from behavior policy
        used for logging (should be instance of TauListDataset)
    :param val_freq: frequency of how often we perform validation logging (only
        if logger object is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractWLogger)
    :return: None
    """
    assert isinstance(f, WAdversaryWrapper)
    assert isinstance(train_data, TauListDataset)
    train_data_loader = train_data.get_data_loader(batch_size)
    if val_data:
        assert isinstance(val_data, TauListDataset)
        val_data_loader = val_data.get_data_loader(batch_size)
    else:
        val_data_loader = None

    for epoch in range(num_epochs):
        if logger and epoch % val_freq == 0:
            logger.log(train_data_loader, val_data_loader, w, f,
                       init_state_sampler, epoch)

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

        if w_scheduler:
            w_scheduler.step()
        if f_scheduler:
            f_scheduler.step()


def train_w_taxi(env, train_data, val_data, pi_e, pi_b, init_state_sampler, logger, gamma, ERM_epoch=50, epoch=10000, w_lr_pre=1e-3,  w_lr=1e-4):
    # define networks and optimizers
    w = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    f = WAdversaryWrapper(StateEmbeddingModel(num_s=env.num_s, num_out=1))
    # w = TaxiSimpleCNN(num_out=1)
    # f = WAdversaryWrapper(TaxiSimpleCNN(num_out=1))

    w_optimizer_pre = Adam(w.parameters(), lr=w_lr_pre, betas=(0.5, 0.9))
    w_optimizer = OAdam(w.parameters(), lr=w_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=w_lr*500, betas=(0.5, 0.9))

    train_w_network_erm(train_data=train_data, pi_e=pi_e, pi_b=pi_b,
                        num_epochs=ERM_epoch, batch_size=1024, w=w,
                        w_optimizer=w_optimizer_pre, gamma=gamma,
                        val_data=val_data, val_freq=10, logger=logger)
    train_w_network(train_data=train_data, pi_e=pi_e, pi_b=pi_b,
                    num_epochs=epoch, batch_size=1024, w=w, f=f,
                    w_optimizer=w_optimizer, f_optimizer=f_optimizer,
                    gamma=gamma, init_state_sampler=init_state_sampler,
                    val_data=val_data, val_freq=10, logger=logger)
    return w


def debug():
    # set up environment and policies
    env = TaxiEnvironment(discrete_state=True)
    # env = CartpoleEnvironment()
    gamma = 0.98
    alpha = 0.6
    temp = 2.0
    hidden_dim = 50
    state_dim = 4
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_s = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)
    # pi_e = load_taxi_policy_continuous(
    #     "taxi_data/saved_policies/pi19.npy", env)
    # pi_s = load_taxi_policy_continuous("taxi_data/saved_policies/pi3.npy", env)
    # pi_b = GenericMixturePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)

    # set up logger
    oracle_tau_len = 100000  # // 10000
    init_state_dist_path = "taxi_data/init_state_dist.npy"
    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)
    # init_state_sampler = DecodingDiscreteInitStateSampler(init_state_dist,
    #                                                       env.decode_state)
    # init_state_sampler = CartpoleInitStateSampler(env)
    # logger = SimpleDiscretePrintWLogger(env=env, pi_e=pi_e, pi_b=pi_b,
    #                                     gamma=gamma,
    #                                     oracle_tau_len=oracle_tau_len)
    logger = DiscreteWLogger(env=env, pi_e=pi_e, pi_b=pi_b, gamma=gamma,
                             tensorboard=True, save_model=True, oracle_tau_len=oracle_tau_len)

    # generate train, val, and test data
    tau_len = 200000  # // 10000
    burn_in = 100000  # // 10000
    train_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                       burn_in=burn_in)
    print('finish train')
    val_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                     burn_in=burn_in)
    print('finish val')
    test_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                      burn_in=burn_in)
    print('finish test')

    w = train_w_taxi(env, train_data, val_data, pi_e, pi_b,
                     init_state_sampler, logger, gamma)
    # calculate final performance
    test_data_loader = test_data.get_data_loader(1024)
    policy_val_est = w_estimator(tau_list_data_loader=test_data_loader,
                                 pi_e=pi_e, pi_b=pi_b, w=w)
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    squared_error = (policy_val_est - policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()
