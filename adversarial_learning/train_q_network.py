from adversarial_learning.game_objectives import q_game_objective
from adversarial_learning.oadam import OAdam
from adversarial_learning.tau_list_dataset import TauListDataLoader
from environments.taxi_environment import TaxiEnvironment
from estimators.benchmark_estimators import on_policy_estimate
from estimators.discrete_estimators import q_estimator_discrete
from models.discrete_models import StateEmbeddingModel
from policies.discrete_policy import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy
from utils.torch_utils import load_tensor_from_npy
from debug_logging.q_logger import SimplePrintQLogger



def train_q_network(train_tau_list, pi_e, num_epochs, batch_size, q, f,
                    q_optimizer, f_optimizer, gamma, val_tau_list=None,
                    val_freq=10, logger=None):
    """
    :param train_tau_list: list of trajectories logged from behavior policy
        to use for training
    :param pi_e: evaluation policy (should be from policies module)
    :param num_epochs: number of epochs to perform training for
    :param batch_size: batch size for alternating gradient descent
    :param q: initial q network (should be torch.nn.Module)
    :param f: initial f network (should be torch.nn.Module)
    :param q_optimizer: optimizer for q network
    :param f_optimizer: optimizer for f network
    :param gamma: discount factor (0 < gamma <= 1)
    :param val_tau_list: (optional) list of validation trajectories for logging
    :param val_freq: frequency of how often we perform validation logging (only
        if logger object is provided)
    :param logger: (optional) logger object (should be subclass of
        AbstractQLogger)
    :return: None
    """
    train_data_loader = TauListDataLoader(tau_list=train_tau_list,
                                          batch_size=batch_size)
    if val_tau_list:
        val_data_loader = TauListDataLoader(tau_list=val_tau_list,
                                            batch_size=batch_size)
    else:
        val_data_loader = None

    for epoch in range(num_epochs):
        for s, a, s_prime, r in train_data_loader:
            q_obj, f_obj = q_game_objective(q, f, s, a, s_prime, r, pi_e, gamma)

            f_optimizer.zero_grad()
            f_obj.backward(retain_graph=True)
            f_optimizer.step()

            q_optimizer.zero_grad()
            q_obj.backward()
            q_optimizer.step()

        if logger and epoch % val_freq == 0:
            logger.log(train_data_loader, val_data_loader, q, f, epoch)


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
    logger = SimplePrintQLogger(env=env, pi_e=pi_e, gamma=gamma,
                                init_state_dist=init_state_dist)

    # generate train and val data
    train_tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=200000,
                                           burn_in=100000)
    val_tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=200000,
                                         burn_in=100000)

    # define networks and optimizers
    q = StateEmbeddingModel(num_s=env.num_s, num_out=env.num_a)
    f = StateEmbeddingModel(num_s=env.num_s, num_out=env.num_a)
    q_lr = 1e-3
    q_optimizer = OAdam(q.parameters(), lr=q_lr, betas=(0.5, 0.9))
    f_optimizer = OAdam(f.parameters(), lr=q_lr*5, betas=(0.5, 0.9))

    train_q_network(train_tau_list=train_tau_list, pi_e=pi_e, num_epochs=1000,
                    batch_size=1024, q=q, f=f, q_optimizer=q_optimizer,
                    f_optimizer=f_optimizer, gamma=gamma,
                    val_tau_list=val_tau_list, val_freq=10, logger=logger)

    # calculate final performance
    policy_val_est = q_estimator_discrete(pi_e=pi_e, gamma=gamma, q=q,
                                          init_state_dist=init_state_dist)
    policy_val_oracle = on_policy_estimate(env=env, pi_e=pi_e, gamma=gamma,
                                           num_tau=1, tau_len=1000000)
    squared_error = (policy_val_est - policy_val_oracle) ** 2
    print("Test policy val squared error:", squared_error)


if __name__ == "__main__":
    debug()

