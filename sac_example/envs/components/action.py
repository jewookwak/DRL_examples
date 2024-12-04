# envs/components/action.py
import numpy as np

class Action:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_dim = action_space.shape[0]
        self.action_bound = action_space.high[0]

    def process(self, action):
        return np.clip(action, -self.action_bound, self.action_bound)