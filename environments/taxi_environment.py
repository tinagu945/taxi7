import numpy as np
import torch
from environments.abstract_environment import AbstractEnvironment
from policies.debug_policies import RandomPolicy

DOWN_ACTION = 0
RIGHT_ACTION = 1
UP_ACTION = 2
LEFT_ACTION = 3


class TaxiEnvironment(AbstractEnvironment):
    def __init__(self, grid_length=5, discrete_state=True):
        self.n_state = 0
        self.n_action = 6
        self.grid_length = grid_length
        self.n_state = (grid_length ** 2) * 16 * 5
        self.possible_passenger_loc = [(0, 0), (0, grid_length - 1),
                                       (grid_length - 1, 0),
                                       (grid_length - 1, grid_length - 1)]
        self.discrete_state = discrete_state
        super().__init__(num_s=self.n_state, num_a=self.n_action,
                         is_s_discrete=discrete_state)
        self.reset()

    def reset(self):
        self._reset_internal()
        if self.discrete_state:
            return self._state_encoding()
        else:
            return self._state_tensor()

    def _reset_internal(self):
        self.x = np.random.randint(self.grid_length)
        self.y = np.random.randint(self.grid_length)
        self.passenger_status = np.random.randint(16)
        self.taxi_status = 4

    def _state_encoding(self):
        length = self.grid_length
        return (self.taxi_status
                + (self.passenger_status + (self.x * length + self.y) * 16) * 5)

    def _state_tensor(self):
        state_tensor = torch.zeros(self.grid_length, self.grid_length, 3)
        # first channel encodes empty taxi position
        # second channel encodes passenger positions
        # third channel encodes taxi destination
        state_tensor[self.x, self.y, 0] = 1.0
        for i in range(4):
            if self.passenger_status & (1 << i):
                x, y = self.possible_passenger_loc[i]
                state_tensor[x, y, 1] = 1.0
        if self.taxi_status < 4:
            x, y = self.possible_passenger_loc[self.taxi_status]
            state_tensor[x, y, 2] = 1.0
        return state_tensor

    def encode_state_tensor(self, s):
        """
        :param s: either single state of shape (grid_length, grid_length, 3),
            or batch of states of size (b, grid_length, grid_length, 3)
        :return: either integer representing decoding of single state, or
            LongTensor of integers representing decoding of state batch
        """
        if len(s.shape) == 4:
            is_batch = True
        elif len(s.shape) == 3:
            is_batch = False
        else:
            raise ValueError("Invalid shape for state input:", s.shape)
        s = s.long()
        c = torch.LongTensor(range(self.grid_length))
        if is_batch:
            batch_size = s.shape[0]
            taxi_x = torch.einsum("bij,i->b", s[:, :, :, 0], c)
            taxi_y = torch.einsum("bij,j->b", s[:, :, :, 0], c)
            passenger_status = torch.zeros(batch_size).long()
            taxi_status = torch.ones(batch_size).long() * 4
            for i in range(4):
                x, y = self.possible_passenger_loc[i]
                passenger_status.add_(s[:, x, y, 1] * (1 << i))
                taxi_status.add_(s[:, x, y, 2] * (i - 4))
        else:
            taxi_x = int(torch.einsum("ij,i->", s[:, :, 0], c))
            taxi_y = int(torch.einsum("ij,j->", s[:, :, 0], c))
            passenger_status = 0
            taxi_status = 4
            for i in range(4):
                x, y = self.possible_passenger_loc[i]
                passenger_status += (int(s[x, y, 1]) * (1 << i))
                taxi_status += int(s[x, y, 2]) * (i - 4)
        return (taxi_status + 5 * (passenger_status +
                                   (taxi_x * self.grid_length + taxi_y) * 16))

    def decode_state(self, s):
        """
        :param s: either single integer state encoding or a LongTensor batch
            of state encodings
        :return: either a single decoded state, or a batch of decoded states
        """
        length = self.grid_length
        taxi_status = s % 5
        passenger_status = (s // 5) % 16
        taxi_y = (s // 80) % length
        taxi_x = (s // 80) // length
        if isinstance(s, torch.Tensor):
            batch_size = s.shape[0]
            batch_idx = torch.LongTensor(range(batch_size))
            s_decoded = torch.zeros(batch_size, length, length, 3)
            s_decoded[batch_idx, taxi_x, taxi_y, 0] = 1.0
            for i in range(4):
                x, y = self.possible_passenger_loc[i]
                passenger_flags = ((passenger_status & (1 << i)) > 0).float()
                s_decoded[batch_idx, x, y, 1] = passenger_flags
                destination_flags = (taxi_status == i).float()
                s_decoded[batch_idx, x, y, 2] = destination_flags
        elif isinstance(s, int):
            s_decoded = torch.zeros(length, length, 3)
            s_decoded[taxi_x, taxi_y, 0] = 1.0
            for i in range(4):
                x, y = self.possible_passenger_loc[i]
                if passenger_status & (1 << i):
                    s_decoded[x, y, 1] = 1.0
                if taxi_status == i:
                    s_decoded[x, y, 2] = 1.0
        else:
            raise ValueError("invalid state or state batch: %r" % s)
        return s_decoded

    def render(self):
        MAP = []
        length = self.grid_length
        for i in range(length):
            if i == 0:
                MAP.append("-" * (3 * length + 1))
            MAP.append("|" + "  |" * length)
            MAP.append("-" * (3 * length + 1))
        MAP = np.asarray(MAP, dtype="c")
        # state 4 is empty
        if self.taxi_status == 4:
            MAP[2 * self.x + 1, 3 * self.y + 2] = "O"
        else:
            # @ means non-empty
            MAP[2 * self.x + 1, 3 * self.y + 2] = "@"
        for i in range(4):
            # a means passenger
            if self.passenger_status & (1 << i):
                x, y = self.possible_passenger_loc[i]
                MAP[2 * x + 1, 3 * y + 1] = "a"
        for line in MAP:
            print(''.join([i.decode('UTF-8') for i in line]))
        if self.taxi_status == 4:
            print('Empty Taxi')
        else:
            x, y = self.possible_passenger_loc[self.taxi_status]
            print('Taxi destination:({},{})'.format(x, y))

    def step(self, action):
        reward = -1
        if action == DOWN_ACTION:
            if self.x < self.grid_length - 1:
                self.x += 1
        elif action == RIGHT_ACTION:
            if self.y < self.grid_length - 1:
                self.y += 1
        elif action == LEFT_ACTION:
            if self.x > 0:
                self.x -= 1
        elif action == UP_ACTION:
            if self.y > 0:
                self.y -= 1
        elif action == 4:  # Try to pick up
            for i in range(4):
                x, y = self.possible_passenger_loc[i]
                if x == self.x and y == self.y and (
                    self.passenger_status & (1 << i)):
                    # successfully pick up
                    self.passenger_status -= 1 << i
                    self.taxi_status = np.random.randint(4)
                    while self.taxi_status == i:
                        self.taxi_status = np.random.randint(4)
        elif action == 5:
            if self.taxi_status < 4:
                x, y = self.possible_passenger_loc[self.taxi_status]
                if self.x == x and self.y == y:
                    # 					print('got reward!')
                    reward = 20
                self.taxi_status = 4
        self._change_passenger_status()
        if self.discrete_state:
            return self._state_encoding(), reward
        else:
            return self._state_tensor(), reward

    def _change_passenger_status(self):
        p_generate = [0.3, 0.05, 0.1, 0.2]
        p_disappear = [0.05, 0.1, 0.1, 0.05]
        for i in range(4):
            if self.passenger_status & (1 << i):
                if np.random.rand() < p_disappear[i]:
                    self.passenger_status -= 1 << i
            else:
                if np.random.rand() < p_generate[i]:
                    self.passenger_status += 1 << i


def debug():
    env = TaxiEnvironment(discrete_state=True)
    pi = RandomPolicy(num_a=env.num_a, state_rank=3)
    tau_list = env.generate_roll_out(pi=pi, num_tau=1, tau_len=10)
    s, a, s_prime, r = tau_list[0]
    print("states:")
    print(s_prime)
    print("decoded states:")
    print(env.decode_state(s_prime))
    # print("encoded states one by one:")
    # for s_ in s_prime:
    #     print(env.encode_state_tensor(s_))
    print("encoded decoded states")
    print(env.encode_state_tensor(env.decode_state(s_prime)))
    print("encoded decoded states one by one")
    for s_ in s_prime:
        s_ = int(s_)
        print(env.encode_state_tensor(env.decode_state(s_)))



if __name__ == "__main__":
    debug()
