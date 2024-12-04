# train/trainer.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pylab
from datetime import datetime

from train.networks import DQN
from train.replay_buffer import ReplayBuffer

class DQNTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Create save directories
        os.makedirs(self.config.WEIGHTS_PATH, exist_ok=True)
        # os.makedirs(self.config.GRAPH_PATH, exist_ok=True)
        
        # Initialize networks
        self.model = DQN(env.action_space.n)
        self.target_model = DQN(env.action_space.n)
        self.optimizer = Adam(learning_rate=self.config.LEARNING_RATE)
        
        # Initialize training variables
        self.epsilon = self.config.EPSILON
        self.buffer = ReplayBuffer(self.config.MEMORY_SIZE)
        self.best_reward = float('-inf')
        
        # Initialize metrics
        self.rewards = []
        self.episodes = []
        self.avg_reward = 0
        
        # TensorBoard setup
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('logs', 'DQN', current_time)
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        # Initialize target network
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])

    def train_step(self, states, actions, rewards, next_states, dones):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.env.action_space.n)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            # 다음 상태에 대한 타깃 모델의 큐함수
            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.config.DISCOUNT_FACTOR * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss

    def train(self):
        for ep in range(self.config.MAX_EPISODES):
            state,_ = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            episode_reward, done = 0, False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                
                # Modify reward: 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
                episode_reward += reward
                
                self.buffer.add_buffer(state, action, reward, next_state, done)
                
                if self.buffer.buffer_count() >= self.config.TRAIN_START:
                    if self.epsilon > self.config.EPSILON_MIN:
                        self.epsilon *= self.config.EPSILON_DECAY
                        
                    # Sample batch and train
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(
                        self.config.BATCH_SIZE)
                    loss = self.train_step(states, actions, rewards, next_states, dones)
                    
                    # Log training metrics
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.buffer.buffer_count())
                        tf.summary.scalar('epsilon', self.epsilon, step=self.buffer.buffer_count())
                
                state = next_state
                
                if done:
                    # Update target network and compute metrics
                    self.update_target_network()
                    self.avg_reward = 0.9 * self.avg_reward + 0.1 * episode_reward if self.avg_reward != 0 else episode_reward
                    
                    print(f"episode: {ep:3d} | "
                          f"episode reward: {episode_reward:3.0f} | "
                          f"average reward: {self.avg_reward:3.2f} | "
                          f"memory length: {self.buffer.buffer_count():4d} | "
                          f"epsilon: {self.epsilon:.4f}")
                    
                    # Log episode metrics
                    with self.summary_writer.as_default():
                        tf.summary.scalar('reward/episode_reward', episode_reward, step=ep)
                        tf.summary.scalar('reward/average_reward', self.avg_reward, step=ep)
                    
                    # # Save training graph
                    # self.rewards.append(self.avg_reward)
                    # self.episodes.append(ep)
                    # pylab.plot(self.episodes, self.rewards, 'b')
                    # pylab.xlabel("episode")
                    # pylab.ylabel("average reward")
                    # pylab.savefig(f"{self.config.GRAPH_PATH}graph.png")
                    
                    # Save weights based on conditions
                    if self.avg_reward > self.config.REWARD_THRESHOLD:
                        self._save_weights(best=True)
                        print("\nProblem solved!")
                        # break
                        return
                    elif episode_reward >= self.best_reward:
                        self.best_reward = episode_reward
                        self._save_weights(best=True)
                    elif ep % self.config.SAVE_INTERVAL == 0:
                        self._save_weights(best=False)

    def _save_weights(self, best=False):
        prefix = "best_" if best else ""
        # 메인 모델 저장
        self.model.save_weights(f"{self.config.WEIGHTS_PATH}{prefix}cartpole_dqn_model.h5")