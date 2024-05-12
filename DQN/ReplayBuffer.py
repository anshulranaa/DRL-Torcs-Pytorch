import numpy as np
import random 

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []

    def add(self, s, a, r, s_next, done):
        experience = (s, a, r, s_next, done)
        if self.count < self.buffer_size:  # Corrected from 'bufferor_size' to 'buffer_size'
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        return self.count

    def getBatch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = [_[0] for _ in batch]
        a_batch = [_[1] for _ in batch]
        r_batch = [_[2] for _ in batch]
        s_next_batch = [_[3] for _ in batch]
        done_batch = [_[4] for _ in batch]

        return s_batch, a_batch, r_batch, s_next_batch, done_batch
