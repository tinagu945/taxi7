import torch

from dataset.tau_list_dataset import TauListDataset
from debug_logging.w_logger import DiscreteWLogger
from environments.taxi_environment import TaxiEnvironment
from models.discrete_models import StateEmbeddingModel
from policies.mixture_policies import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy


def erm_w_objective(w, s, a, s_prime, pi_e, pi_b, gamma):
    w_of_s = w(s).view(-1)
    w_of_s_prime = w(s_prime).view(-1)
    pi_ratio = pi_e(s) / pi_b(s)
    eta = torch.gather(pi_ratio, dim=1, index=a.view(-1, 1)).view(-1)
    epsilon = gamma * w_of_s * eta - w_of_s_prime + 1 - gamma
    main_obj = (epsilon ** 2).mean()
    reg_1 = (w_of_s.mean() - 1.0) ** 2
    w_of_s_neg = (w_of_s < 0).float()
    reg_2 = ((w_of_s * w_of_s_neg) ** 2).mean()
    return main_obj + 0.1 * reg_1 + 0.1 * reg_2


def train_w_network_erm(train_data, pi_e, pi_b, num_epochs,
                        batch_size, w, w_optimizer, gamma,
                        val_data=None, val_freq=10, logger=None):
    """
    :param train_data: training data logged from behavior policy (should be
        an instance of TauListDataset)
    :param pi_e: evaluation policy (should be from policies module)
    :param pi_e: behavior policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param w: initial w network (should be torch.nn.Module)
    :param w_optimizer: optimizer for w network
    :param gamma: discount factor (0 < gamma <= 1)
    :param val_data: (optional) validation data for logging (if provided, should
        be an instance of TauListDataset)
    :param val_freq: frequency of how often we perform validation logging (only
        if validation data is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractQLogger)
    :return: None
    """
    assert isinstance(train_data, TauListDataset)
    train_data_loader = train_data.get_data_loader(batch_size)
    if val_data:
        assert isinstance(val_data, TauListDataset)
        val_data_loader = val_data.get_data_loader(batch_size)
    else:
        val_data_loader = None

    for epoch in range(num_epochs):
        for s, a, s_prime, r in train_data_loader:
            obj = erm_w_objective(w, s, a, s_prime, pi_e, pi_b, gamma)
            w_optimizer.zero_grad()
            obj.backward()
            w_optimizer.step()

        if logger and epoch % val_freq == 0:
            logger.log_benchmark(train_data_loader, val_data_loader, w, epoch)


def train_w_network_erm_lbfgs(train_data, pi_e, pi_b, w, gamma,
                              val_data=None, logger=None):
    w_optimizer = torch.optim.LBFGS(w.parameters())
    s, a, s_prime, r = (train_data.s, train_data.a,
                        train_data.s_prime, train_data.r)

    def closure():
        w_optimizer.zero_grad()
        obj = erm_w_objective(w, s, a, s_prime, pi_e, pi_b, gamma)
        obj.backward()
        return obj

    w_optimizer.step(closure)

    if logger:
        train_data_loader = train_data.get_data_loader(1024)
        val_data_loader = val_data.get_data_loader(1024)
        logger.log_benchmark(train_data_loader, val_data_loader, w, 0)


def debug():
    env = TaxiEnvironment()
    gamma = 0.98
    alpha = 0.6
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_s = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_s, pi_1_weight=alpha)

    # set up logger
    oracle_tau_len = 1000000
    logger = DiscreteWLogger(env=env, pi_e=pi_e, pi_b=pi_b,
                             gamma=gamma, tensorboard=False, save_model=False,
                             oracle_tau_len=oracle_tau_len)

    # generate train, val, and test data
    tau_len = 200000
    burn_in = 100000
    train_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                       burn_in=burn_in)
    val_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=tau_len,
                                     burn_in=burn_in)

    # define networks and optimizers
    w = StateEmbeddingModel(num_s=env.num_s, num_out=1)
    w_lr = 1e-3
    w_optimizer = torch.optim.Adam(w.parameters(), lr=w_lr, betas=(0.5, 0.9))

    train_w_network_erm_lbfgs(train_data=train_data, pi_e=pi_e, pi_b=pi_b, w=w,
                              gamma=gamma, val_data=val_data, logger=logger)


if __name__ == "__main__":
    debug()
