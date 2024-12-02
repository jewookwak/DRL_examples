# train/config.py
class Config:
    def __init__(self):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.MAX_EPISODES = 200
        self.MIN_BUFFER_SIZE = 1000
        self.WEIGHTS_PATH = "./save_weights/" 