# train/replay_buffer.py
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        
    def add_buffer(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])
        
        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer.clear()