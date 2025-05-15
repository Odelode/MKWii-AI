import random
import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: np.array(x, dtype=np.float32), zip(*batch)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)