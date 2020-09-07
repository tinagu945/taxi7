from torch.optim import Adam
from adversarial_learning.game_objectives import q_game_objective
from adversarial_learning.oadam import OAdam
from dataset.init_state_sampler import DiscreteInitStateSampler
from dataset.tau_list_dataset import TauListDataset
from benchmark_methods.discrete_q_benchmark import fit_q_tabular
from environments.taxi_environment import TaxiEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.infinite_horizon_estimators import q_estimator
from models.discrete_models import StateEmbeddingModel
from policies.mixture_policies import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy
from utils.torch_utils import load_tensor_from_npy
from debug_logging.q_logger import DiscreteQLogger


def train_q_network(train_data, pi_e, num_epochs, batch_size, q,
                    f, q_optimizer, f_optimizer, gamma, val_data=None,
                    val_freq=10, logger=None):
    """
    :param train_data: dataset logged from behavior policy used for training
        (should be instance of TauListDataset)
    :param pi_e: evaluation policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param q: initial q network (should be torch.nn.Module)
    :param f: initial f network (should be torch.nn.Module)
    :param q_optimizer: optimizer for q network
    :param f_optimizer: optimizer for f network
    :param gamma: discount factor (0 < gamma <= 1)
    :param val_data: (optional) additional dataset logged from behavior policy
        used for logging (should be instance of TauListDataset)
    :param val_freq: frequency (in terms of epochs) of how often we perform
        validation logging (only if logger object is provided)
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
        print(epoch)
        if logger and epoch % val_freq == 0:
            logger.log(train_data_loader, val_data_loader, q, f, epoch)

        for s, a, s_prime, r in train_data_loader:
            q_obj, f_obj = q_game_objective(
                q, f, s, a, s_prime, r, pi_e, gamma)

            f_optimizer.zero_grad()
            f_obj.backward(retain_graph=True)
            f_optimizer.step()

            q_optimizer.zero_grad()
            q_obj.backward()
            q_optimizer.step()


def debug():
    # set up environment and policies
    env = TaxiEnvironment()
    gamma = 0.98
    alpha = 0.6
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_other = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_other, pi_1_weight=alpha)

    # set up logger
    init_state_dist_path = "taxi_data/init_state_dist.npy"
    init_state_dist = load_tensor_from_npy(init_state_dist_path).view(-1)
    init_state_sampler = DiscreteInitStateSampler(init_state_dist)
    logger = DiscreteQLogger(env=env, pi_e=pi_e, gamma=gamma, tensorboard=True,
                             save_model=True, init_state_sampler=init_state_sampler)
    # generate train and val data
    train_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=200000,
                                       burn_in=100000)
    val_data = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=200000,
                                     burn_in=100000)
    # define networks and optimizers
    # q = StateEmbeddingModel(num_s=env.num_s, num_out=env.num_a)
    q = fit_q_tabular(data=train_data, pi=pi_e, gamma=gamma)
    f = StateEmbeddingModel(num_s=env.num_s, num_out=env.num_a)
    q_pretrain_lr = 1e-1
    q_pretrain_optimizer = Adam(q.parameters(), lr=q_pretrain_lr)
    q_lr = 1e-3
    q_optimizer = OAdam(q.parameters(), lr=q_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=q_lr*5, betas=(0.5, 0.9))
    # do ERM pre-training
    # train_q_network_erm(train_tau_list=train_tau_list, pi_e=pi_e,
    #                     num_epochs=100, batch_size=1024, q=q,
    #                     q_optimizer=q_pretrain_optimizer, gamma=gamma,
    #                     val_tau_list=val_tau_list, val_freq=10, logger=logger)

    # train using adversarial algorithm
    train_q_network(train_data=train_data, pi_e=pi_e, num_epochs=1000,
                    batch_size=1024, q=q, f=f, q_optimizer=q_optimizer,
                    f_optimizer=f_optimizer, gamma=gamma,
                    val_data=val_data, val_freq=10, logger=logger)

    # calculate final performance
    policy_val_est = q_estimator(pi_e=pi_e, gamma=gamma, q=q,
                                 init_state_sampler=init_state_sampler)
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    squared_error = (policy_val_est - policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()
