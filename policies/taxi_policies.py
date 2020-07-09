import numpy as np
import torch
from environments.taxi_environment import TaxiEnvironment
from policies.discrete_policy import DiscretePolicy, DecodingDiscretePolicy


def load_taxi_policy(path):
    pi_table = np.load(path)
    return DiscretePolicy(pi_table)


def load_taxi_policy_continuous(path, env):
    assert isinstance(env, TaxiEnvironment)
    pi_table = np.load(path)
    return DecodingDiscretePolicy(pi_table, env.decode_state_tensor)


def debug():
    env = TaxiEnvironment(discrete_state=False)
    pi = load_taxi_policy_continuous("taxi_data/saved_policies/pi19.npy", env)
    tau_list = env.generate_roll_out(pi=pi, num_tau=1, tau_len=10)
    s, a, s_prime, r = tau_list[0]
    print("policy probs")
    print(pi.policy_probs)
    print("")
    print("policy probs on batch")
    policy_probs = pi(s)
    print(policy_probs)
    a_probs = torch.gather(policy_probs, 1, a.view(-1, 1)).view(-1)
    print("probs of selected actions:", a_probs)
    print("mean prob:", a_probs.mean())


if __name__ == "__main__":
    debug()
