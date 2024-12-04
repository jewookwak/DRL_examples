# envs/pendulum_env.py
from envs.base_env import BaseEnv
from envs.components.state import State
from envs.components.action import Action
from envs.components.reward import Reward
import gym
class PendulumEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.env = gym.make("Pendulum-v0")
        self.state_handler = State(self.env.observation_space)
        self.action_handler = Action(self.env.action_space)
        self.reward_handler = Reward()

    def reset(self):
        state = self.env.reset()
        return self.state_handler.process(state)

    def step(self, action):
        processed_action = self.action_handler.process(action)
        next_state, reward, done, info = self.env.step(processed_action)
        processed_state = self.state_handler.process(next_state)
        processed_reward = self.reward_handler.process(reward)
        return processed_state, processed_reward, done, info

    def close(self):
        self.env.close()
