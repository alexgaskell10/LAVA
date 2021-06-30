import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity=1e5):
        self._capacity = capacity
        self._empty_deque()

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def _empty_deque(self):
        self.memory = deque([], maxlen=int(self._capacity))


'''
- Whole sample (get sampler index somehow?!?)
- Retrievals and order of retrievals
'''