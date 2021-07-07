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
        self.empty()

    def push(self, sample):
        '''Save a transition'''
        self.memory.append(sample)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def empty(self):
        self.memory = deque([], maxlen=int(self._capacity))

    def peek(self, idx=0):
        return self.memory[idx]

    def __iter__(self):
        yield from self.memory
