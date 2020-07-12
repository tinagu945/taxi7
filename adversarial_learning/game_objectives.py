import torch
from environments.taxi_environment import TaxiEnvironment
from models.w_adversary_wrapper import WAdversaryWrapper
from models.discrete_models import StateEmbeddingModel
from policies.discrete_policy import MixtureDiscretePolicy
from policies.taxi_policies import load_taxi_policy


def q_game_objective(q, f, s, a, s_prime, r, pi_e, gamma):
    """
    :param q: q function, should be nn.Module object that takes batch of
        states, and returns tensor of shape (batch_size, num_actions)
    :param f: critic function, should be nn.Module object that takes batch
        of states, and returns tensor of shape (batch_size, num_actions)
    :param s: batch of states, should be shape (batch_size, *s_shape)
    :param a: batch of actions, should be a LongTensor of shape (batch_size,)
    :param s_prime: batch of successor states, should be shape
        (batch_size, *s_shape)
    :param r: batch of rewards, should be shape (batch_size,)
    :param pi_e: evaluation policy, given batch s_ of states of shape
        (b, *s_shape), pi_e(s_) should return corresponding batch of action
        probabilities of shape (b, num_actions)
    :param gamma: discount factor, should satisfy 0.0 < gamma <= 1.0
    :return: tuple of the form (q_obj, f_obj), where q_obj is the objective
        for q to minimize using SGD, and f_obj is the objective for f to
        minimize
    """
    #q_of_s_a: (batch_size,)
    q_of_s_a = torch.gather(q(s), dim=1, index=a.view(-1, 1)).view(-1)
    f_of_s_a = torch.gather(f(s), dim=1, index=a.view(-1, 1)).view(-1)
    v_of_ss = (pi_e(s_prime) * q(s_prime)).sum(1)
    
    
    #TODO: implement QWrapper
    #c = q.get_constraint_multipliers()
    c=1.0
    constraint = q_of_s_a - ((pi_e(s_prime) * r.unsqueeze(-1)).sum(1)+gamma*v_of_ss)
      
    m = (r + gamma * v_of_ss - q_of_s_a) * f_of_s_a +c*constraint
    moment = m.mean()
    f_reg = (m ** 2).mean()
    return moment, -moment + 0.25 * f_reg


def w_game_objective(w, f, s, a, s_prime, pi_e, pi_b, s_0, gamma):
    """
    :param w: w function, should be nn.Module object that takes batch of
        states, and returns tensor of shape (batch_size,)
    :param f: critic function, should be WAdversaryWrapper object that
    takes batch
        of states, and returns tensor of shape (batch_size,)
    :param s: batch of states, should be shape (batch_size, *s_shape)
    :param a: batch of actions, should be a LongTensor of shape (batch_size,)
    :param s_prime: batch of successor states, should be shape
        (batch_size, *s_shape)
    :param pi_e: evaluation policy, should be from policies module
    :param pi_b: behavior policy, should be from policies module
    :param s_0: batch of initial states, should be shape (batch_size, *s_shape)
    :param gamma: discount factor, should satisfy 0.0 < gamma <= 1.0
    :return: tuple of the form (w_obj, f_obj), where w_obj is the objective
        for q to minimize using SGD, and f_obj is the objective for f to
        minimize
    """
    assert isinstance(f, WAdversaryWrapper)
    w_of_s = w(s).view(-1)
    w_of_s_prime = w(s_prime).view(-1)
    f_of_s_prime = f(s_prime).view(-1)
    f_of_s_0 = f(s_0).view(-1)
    pi_ratio = pi_e(s) / pi_b(s)        
    eta_s_a = torch.gather(pi_ratio, dim=1, index=a.view(-1, 1)).view(-1)

#     import pdb;pdb.set_trace()
    epsilon = gamma * w_of_s * eta_s_a - w_of_s_prime
    c = f.get_constraint_multipliers()
    m_1 = (f_of_s_prime * epsilon
           + c[0] * (w_of_s - 1.0)
           + c[1] * (w_of_s_prime - 1.0))
    m_2 = (1 - gamma) * f_of_s_0
    moment = m_1.mean() + m_2.mean()
    f_reg = (m_1 ** 2).mean() + (m_2 ** 2).mean() + 2 * m_1.mean() * m_2.mean()
    return moment, -moment + 0.25 * f_reg


def debug():
    env = TaxiEnvironment()
    gamma = 0.98
    alpha = 0.6
    pi_e = load_taxi_policy("taxi_data/saved_policies/pi19.npy")
    pi_other = load_taxi_policy("taxi_data/saved_policies/pi3.npy")
    pi_b = MixtureDiscretePolicy(pi_1=pi_e, pi_2=pi_other, pi_1_weight=alpha)
    q = StateEmbeddingModel(num_s=env.num_s, num_out=env.num_a)
    f = StateEmbeddingModel(num_s=env.num_s, num_out=env.num_a)

    # generate train and val data
    tau_list = env.generate_roll_out(pi=pi_b, num_tau=1, tau_len=4)
    s, a, s_prime, r = tau_list[0]
    print("s:", s)
    print("a:", a)
    print("s_prime:", s_prime)
    print("r:", r)
    print("")
    q_obj, f_obj = q_game_objective(q, f, s, a, s_prime, r, pi_e, gamma)
    print("q_obj:", q_obj)
    print("f_obj:", f_obj)


if __name__ == "__main__":
    debug()
