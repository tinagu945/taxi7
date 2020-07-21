import torch
from environments.abstract_environment import AbstractEnvironment
from policies.debug_policies import RandomPolicy
import gym


class CartpoleEnvironment(AbstractEnvironment):
    def __init__(self):
        """Only continuous version now"""
        self.gym_env = gym.envs.make("CartPole-v1")
        super().__init__(num_s=float("inf"), num_a=2, is_s_discrete=False)
        self.gym_env.reset()
        self.at_trajectory_end = False
        self.state_dim=4
        
    def reset(self):
        s = torch.from_numpy(self.gym_env.reset()).float()
        return s

    def step(self, action):
        next_state, reward, done, _ = self.gym_env.step(action)
        if done:
            next_state = self.gym_env.reset()
        return torch.from_numpy(next_state).float(), reward


def debug():
    env = CartpoleEnvironment()
    pi = RandomPolicy(num_a=env.num_a, state_rank=3)
    tau_list = env.generate_roll_out(pi=pi, num_tau=1, tau_len=10)
    s, a, s_prime, r = tau_list[0]
    print(s.shape)
    print(a.shape)
    print(s_prime.shape)
    print(r.shape)



if __name__ == "__main__":
    debug()
