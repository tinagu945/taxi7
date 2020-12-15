import torch
from environments.abstract_environment import AbstractEnvironment
from policies.debug_policies import RandomPolicy
import gym


class CartpoleEnvironment(AbstractEnvironment):
    def __init__(self, reward_reshape=False, potential_func=lambda state: -1*((abs(state[0])-2.4)**2+(abs(state[2])-0.20944)**2), gamma=0.98):
        """Only continuous version now
        Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
        Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.

        """
        self.gym_env = gym.envs.make("CartPole-v1").unwrapped
        super().__init__(num_s=float("inf"), num_a=2, is_s_discrete=False)
        self.gym_env.reset()
        self.at_trajectory_end = False
        self.state_dim = 4

        self.potential_func = potential_func
        self.reward_reshape = reward_reshape
        self.gamma = gamma
        if self.reward_reshape:
            assert self.gamma and self.potential_func, "To use reward reshaping you must provide a potential function and a gamma!"

    def reset(self):
        s = torch.from_numpy(self.gym_env.reset()).float()
        return s

    def step(self, action):
        current_state = self.gym_env.state
        next_state, reward, done, _ = self.gym_env.step(action)
        if done:
            next_state = self.gym_env.reset()
            reward = -100000.0

        if self.reward_reshape:
            reward = reward + self.gamma * \
                self.potential_func(next_state) - \
                self.potential_func(current_state)
        return torch.from_numpy(next_state).float(), reward, done


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
