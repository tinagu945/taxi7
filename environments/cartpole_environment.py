import sys
sys.path.insert(0,'./')
import numpy as np
import torch
from environments.abstract_environment import AbstractEnvironment
from policies.debug_policies import RandomPolicy
import gym

class CartpoleEnvironment(AbstractEnvironment):
    def __init__(self):
        """Only continuous version now"""
        self.env = gym.envs.make("CartPole-v1")
        #TODO: num_s number should be inf.
        super().__init__(num_s=4, num_a=2,
                         is_s_discrete=False)
        self.env.reset()

    def reset(self):
        return torch.Tensor([self.env.reset()])

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        return torch.Tensor([next_state]), torch.Tensor([reward])

def debug():
    env = CartpoleEnvironment()
    pi = RandomPolicy(num_a=env.num_a, state_rank=3)
    tau_list = env.generate_roll_out(pi=pi, num_tau=1, tau_len=4)
    s, a, s_prime, r = tau_list[0]
    print(s.shape)
    print(a.shape)
    print(s_prime.shape)
    print(r.shape)



if __name__ == "__main__":
    debug()
