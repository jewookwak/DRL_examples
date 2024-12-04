# train/networks.py
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform

class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.fc_out = Dense(action_size, 
                          kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q