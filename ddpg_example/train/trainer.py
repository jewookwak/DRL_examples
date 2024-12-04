# train/trainer.py
import numpy as np
import tensorflow as tf
from train.networks import Actor, Critic
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from train.replay_buffer import ReplayBuffer
import os
from datetime import datetime

class DDPGTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Create weights directory if it doesn't exist
        os.makedirs(self.config.WEIGHTS_PATH, exist_ok=True)
        
        # Networks
        self.actor = Actor(env.action_handler.action_dim, env.action_handler.action_bound)
        self.target_actor = Actor(env.action_handler.action_dim, env.action_handler.action_bound)
        self.critic = Critic()
        self.target_critic = Critic()
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = Adam(config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(config.CRITIC_LEARNING_RATE)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(config.BUFFER_SIZE)
        
        # Initialize noise
        self.prev_noise = np.zeros(env.action_handler.action_dim)
        
        # TensorBoard setup
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('logs', 'DDPG', current_time)
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        self.save_epi_reward = []
        self.best_reward = float('-inf')
        self.save_interval = 20  # 20 에피소드마다 저장
        self.score_avg = 0
        self.reward_threshold = 180

    def _build_networks(self):
        state_dim = self.env.state_handler.state_dim
        action_dim = self.env.action_handler.action_dim
        
        self.actor.build(input_shape=(None, state_dim))
        self.target_actor.build(input_shape=(None, state_dim))
        
        state_in = Input((state_dim,))
        action_in = Input((action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

    def train(self):
        self.update_target_network(1.0)
        score_avg = 0  # score_avg 초기화

        for ep in range(self.config.MAX_EPISODES):
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                action = self._get_action(state, self.prev_noise)
                next_state, reward, done, _ = self.env.step(action)
                
                self.buffer.add_buffer(state, action, reward, next_state, done)

                if self.buffer.buffer_count() > self.config.MIN_BUFFER_SIZE:
                    self._train_step()

                self.prev_noise = self._calculate_noise(self.prev_noise)
                state = next_state
                episode_reward += reward
                time += 1

            # Calculate average score
            score_avg = 0.9 * score_avg + 0.1 * episode_reward if score_avg != 0 else episode_reward
            print(f'Episode: {ep+1}, Time: {time}, Reward: {episode_reward:.2f}, Average: {score_avg:.2f}')

            # TensorBoard에 reward와 average reward 기록
            with self.summary_writer.as_default():
                tf.summary.scalar('reward/episode_reward', episode_reward, step=ep)
                tf.summary.scalar('reward/average_reward', score_avg, step=ep)
                # tf.summary.scalar('episode_length', time, step=ep)
                self.summary_writer.flush()  # 강제 플러시

            self.save_epi_reward.append(episode_reward)
            

            # 가중치 저장 조건
            if score_avg > self.reward_threshold:
                self._save_weights(best=True)
            elif episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_weights(best=True)
            elif ep % self.save_interval == 0:  # 주기적 저장
                self._save_weights(best=False)

    def _get_action(self, state, pre_noise):
        action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
        action = action.numpy()[0]
        noise = self._calculate_noise(pre_noise)
        return self.env.action_handler.process(action + noise)

    def _calculate_noise(self, pre_noise, rho=0.15, mu=0, dt=1e-1, sigma=0.2):
        return pre_noise + rho*(mu - pre_noise)*dt + sigma*np.sqrt(dt)*np.random.normal(size=self.env.action_handler.action_dim)

    def _train_step(self):
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.config.BATCH_SIZE)
        
        target_actions = self.target_actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
        target_q_values = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32), target_actions])
        td_targets = self._compute_td_targets(rewards, target_q_values.numpy(), dones)
        
        self._update_critic(states, actions, td_targets)
        self._update_actor(states)
        self.update_target_network(self.config.TAU)

    def _compute_td_targets(self, rewards, q_values, dones):
        targets = np.array(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.config.GAMMA * q_values[i]
        return targets

    def _update_critic(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q_values = self.critic([tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(actions, dtype=tf.float32)])
            critic_loss = tf.reduce_mean(tf.square(q_values - td_targets))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # TensorBoard에 critic loss 기록
        with self.summary_writer.as_default():
            tf.summary.scalar('loss/critic_loss', critic_loss, step=self.buffer.buffer_count())
            self.summary_writer.flush()  # 강제 플러시
    def _update_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(tf.convert_to_tensor(states, dtype=tf.float32))
            critic_value = self.critic([tf.convert_to_tensor(states, dtype=tf.float32), actions])
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # TensorBoard에 actor loss 기록
        with self.summary_writer.as_default():
            tf.summary.scalar('loss/actor_loss', actor_loss, step=self.buffer.buffer_count())
            self.summary_writer.flush()  # 강제 플러시
    def update_target_network(self, tau):
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    def _save_weights(self, best=False):
        if best:
            prefix = "best_"
        else:
            prefix = ""
        self.actor.save_weights(f"{self.config.WEIGHTS_PATH}{prefix}pendulum_actor.h5")
        self.critic.save_weights(f"{self.config.WEIGHTS_PATH}{prefix}pendulum_critic.h5")
        if best:
            np.savetxt(f"{self.config.WEIGHTS_PATH}pendulum_epi_reward.txt", self.save_epi_reward)