# train/config.py
class Config:
    def __init__(self):
        # DQN hyperparameters
        self.DISCOUNT_FACTOR = 0.99
        self.LEARNING_RATE = 0.001
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.999
        self.EPSILON_MIN = 0.01
        self.BATCH_SIZE = 64
        self.TRAIN_START = 1000
        
        # Training settings
        self.MAX_EPISODES = 300
        self.MEMORY_SIZE = 2000
        self.WEIGHTS_PATH = "./save_weights/"
        # self.GRAPH_PATH = "./save_graph/"
        
        # Early stopping settings
        self.REWARD_THRESHOLD = 450
        self.SAVE_INTERVAL = 20