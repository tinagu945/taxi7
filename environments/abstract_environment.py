import torch
import numpy as np
from dataset.tau_list_dataset import TauListDataset


class AbstractEnvironment(object):
    def __init__(self, num_s, num_a, is_s_discrete):
        self.num_s = num_s
        self.num_a = num_a
        self.is_s_discrete = is_s_discrete

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
        self.tau_list = []
        for _ in range(num_tau):
            states = []
            actions = []
            rewards = []
            successor_states = []
            for i in range(tau_len + burn_in):
                if i == 0 or (gamma and np.random.rand() > gamma):
                    s = self.reset()
                else:
                    s = successor_states[-1]
                p = np.array(pi(s))
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
            self.tau_list.append((s_tensor, a_tensor, ss_tensor, r_tensor))
        return TauListDataset(self.tau_list)

    def reset(self):
        raise NotImplementedError()

    def step(self, a):
        raise NotImplementedError()
