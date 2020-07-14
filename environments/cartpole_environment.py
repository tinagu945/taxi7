import numpy as np
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
        
    def generate_roll_out(self, pi, num_tau, tau_len, burn_in=0, gamma=None):
        """
        Generates roll out (list of trajectories, each of which is a tuple
            (s, a, s', r) of pytorch arrays
        :param pi: policy for rollout, semantics are for any state s,
            pi(s) should return array of action probabilities
        :param num_tau: number of trajectories to roll out
        :param tau_len: length of each roll out trajectory
        :param burn_in: number of actions to perform at the start of each
            trajectory before we begin logging
        :param gamma: (optional) if provided, at each time step with probability
            (1 - gamma) we reset to an initial state
        :return:
        """
        tau_list = []
        done = None
        for _ in range(num_tau):
            states = []
            actions = []
            rewards = []
            successor_states = []
            for i in range(tau_len + burn_in):
                if i == 0 or (gamma and np.random.rand() > gamma) or done:
                    s = self.reset()
                else:
                    s = successor_states[-1]

                p = (pi(s)).detach().cpu().numpy()          
                p = p / p.sum()
                a = np.random.choice(list(range(self.num_a)), p=p)
                s_prime, r, done = self.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                successor_states.append(s_prime)
            if self.is_s_discrete:
                s_tensor = torch.LongTensor(states[burn_in:])
                ss_tensor = torch.LongTensor(successor_states[burn_in:])
            else:
                s_tensor = torch.stack(states[burn_in:])
                ss_tensor = torch.stack(successor_states[burn_in:])
            a_tensor = torch.LongTensor(actions[burn_in:])
            r_tensor = torch.FloatTensor(rewards[burn_in:])
#             import pdb;pdb.set_trace()
            tau_list.append((s_tensor, a_tensor, ss_tensor, r_tensor))
        return tau_list

    def reset(self):
        return torch.Tensor([self.gym_env.reset()]).squeeze()

    def step(self, action):
        next_state, reward, done, _ = self.gym_env.step(action)
        return torch.Tensor([next_state]).squeeze(), torch.Tensor([reward]).squeeze(), done


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
