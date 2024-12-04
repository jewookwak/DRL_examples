# train/trainer.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import os
from datetime import datetime

from train.networks import Actor, Critic
from train.replay_buffer import ReplayBuffer

class SACTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Create weights directory if it doesn't exist
        os.makedirs(self.config.WEIGHTS_PATH, exist_ok=True)
        
        # Initialize networks
        self.actor = Actor(env.action_handler.action_dim, env.action_handler.action_bound)
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        
        # Build networks
        self._build_networks()
        
        # Initialize optimizers
        self.actor_opt = Adam(self.config.ACTOR_LEARNING_RATE)
        self.critic_1_opt = Adam(self.config.CRITIC_LEARNING_RATE)
        self.critic_2_opt = Adam(self.config.CRITIC_LEARNING_RATE)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.config.BUFFER_SIZE)
        
        # Setup TensorBoard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join('logs', 'SAC', current_time)
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        # Initialize training variables
        self.save_epi_reward = []
        self.best_reward = float('-inf')
        self.score_avg = 0

    def _build_networks(self):
        state_dim = self.env.state_handler.state_dim
        action_dim = self.env.action_handler.action_dim
        
        self.actor.build(input_shape=(None, state_dim))
        
        state_in = Input((state_dim,))
        action_in = Input((action_dim,))
        self.critic_1([state_in, action_in])
        self.critic_2([state_in, action_in])
        self.target_critic_1([state_in, action_in])
        self.target_critic_2([state_in, action_in])
        
    def _get_action(self, state):
        tf_state = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, std = self.actor(tf_state)
        action, _ = self.actor.sample_normal(mu, std)
        return action.numpy()[0]

    def _update_target_network(self, tau):
        # Update critic 1
        phi_1 = self.critic_1.get_weights()
        target_phi_1 = self.target_critic_1.get_weights()
        for i in range(len(phi_1)):
            target_phi_1[i] = tau * phi_1[i] + (1 - tau) * target_phi_1[i]
        self.target_critic_1.set_weights(target_phi_1)
        
        # Update critic 2
        phi_2 = self.critic_2.get_weights()
        target_phi_2 = self.target_critic_2.get_weights()
        for i in range(len(phi_2)):
            target_phi_2[i] = tau * phi_2[i] + (1 - tau) * target_phi_2[i]
        self.target_critic_2.set_weights(target_phi_2)

    def _critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q_1 = self.critic_1([states, actions], training=True)
            loss_1 = tf.reduce_mean(tf.square(q_1-td_targets))

        grads_1 = tape.gradient(loss_1, self.critic_1.trainable_variables)
        self.critic_1_opt.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))

        with tf.GradientTape() as tape:
            q_2 = self.critic_2([states, actions], training=True)
            loss_2 = tf.reduce_mean(tf.square(q_2-td_targets))

        grads_2 = tape.gradient(loss_2, self.critic_2.trainable_variables)
        self.critic_2_opt.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))

        return loss_1, loss_2

    def _actor_learn(self, states):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            actions, log_pdfs = self.actor.sample_normal(mu, std)
            log_pdfs = tf.squeeze(log_pdfs, 1)
            soft_q_1 = self.critic_1([states, actions])
            soft_q_2 = self.critic_2([states, actions])
            soft_q = tf.math.minimum(soft_q_1, soft_q_2)
            loss = tf.reduce_mean(self.config.ALPHA * log_pdfs - soft_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss

    def _compute_td_targets(self, rewards, next_states, dones):
        next_mu, next_std = self.actor(next_states)
        next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)

        target_qs_1 = self.target_critic_1([next_states, next_actions])
        target_qs_2 = self.target_critic_2([next_states, next_actions])
        target_qs = tf.math.minimum(target_qs_1, target_qs_2)

        target_qi = target_qs - self.config.ALPHA * next_log_pdf
        target_qi = target_qi.numpy()

        y_k = np.asarray(target_qi)
        for i in range(target_qi.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.config.GAMMA * target_qi[i]
        return y_k

    def train(self):
        # Initialize target networks
        self._update_target_network(1.0)
        
        for ep in range(self.config.MAX_EPISODES):
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # Get action
                action = self._get_action(state)
                # Clip action to bounds
                action = np.clip(action, -self.env.action_handler.action_bound, 
                               self.env.action_handler.action_bound)
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)
                # Process reward for training
                train_reward = self.env.reward_handler.process(reward)
                
                # Store experience in buffer
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > self.config.MIN_BUFFER_SIZE:
                    # Sample batch from buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(
                        self.config.BATCH_SIZE)
                    
                    # Convert to tensors
                    states = tf.convert_to_tensor(states, dtype=tf.float32)
                    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                    
                    # Compute TD targets
                    td_targets = self._compute_td_targets(rewards, next_states, dones)
                    td_targets = tf.convert_to_tensor(td_targets, dtype=tf.float32)
                    
                    # Update critics
                    critic_loss_1, critic_loss_2 = self._critic_learn(states, actions, td_targets)
                    # Update actor
                    actor_loss = self._actor_learn(states)
                    # Update target networks
                    self._update_target_network(self.config.TAU)
                    
                    # Log losses
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss/critic_1_loss', critic_loss_1, 
                                        step=self.buffer.buffer_count())
                        tf.summary.scalar('loss/critic_2_loss', critic_loss_2, 
                                        step=self.buffer.buffer_count())
                        tf.summary.scalar('loss/actor_loss', actor_loss, 
                                        step=self.buffer.buffer_count())

                state = next_state
                episode_reward += reward
                time += 1

            # Update average score
            self.score_avg = 0.9 * self.score_avg + 0.1 * episode_reward if self.score_avg != 0 else episode_reward
            
            # Log episode results
            print(f'Episode: {ep+1}/{self.config.MAX_EPISODES}, '
                  f'Time: {time}, Reward: {episode_reward:.2f}, '
                  f'Average: {self.score_avg:.2f}')
            
            with self.summary_writer.as_default():
                tf.summary.scalar('reward/episode_reward', episode_reward, step=ep)
                tf.summary.scalar('reward/average_reward', self.score_avg, step=ep)
                self.summary_writer.flush()

            self.save_epi_reward.append(episode_reward)
            
            # Save weights based on conditions
            if self.score_avg > self.config.REWARD_THRESHOLD:
                self._save_weights(best=True)
            elif episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self._save_weights(best=True)
            elif ep % self.config.SAVE_INTERVAL == 0:
                self._save_weights(best=False)

        # Save final results
        np.savetxt(f"{self.config.WEIGHTS_PATH}sac_epi_reward.txt", self.save_epi_reward)
        print(f"Training finished. Final average reward: {self.score_avg:.2f}")

    def _save_weights(self, best=False):
        """Save the network weights."""
        prefix = "best_" if best else ""
        self.actor.save_weights(f"{self.config.WEIGHTS_PATH}{prefix}sac_actor.h5")
        self.critic_1.save_weights(f"{self.config.WEIGHTS_PATH}{prefix}sac_critic_1.h5")
        self.critic_2.save_weights(f"{self.config.WEIGHTS_PATH}{prefix}sac_critic_2.h5")
        if best:
            np.savetxt(f"{self.config.WEIGHTS_PATH}sac_epi_reward.txt", self.save_epi_reward)

    def load_weights(self, path):
        """Load the network weights."""
        try:
            self.actor.load_weights(path + 'best_sac_actor.h5')
            self.critic_1.load_weights(path + 'best_sac_critic_1.h5')
            self.critic_2.load_weights(path + 'best_sac_critic_2.h5')
            print("Successfully loaded best model weights")
        except:
            print("Could not find best weights, trying latest weights...")
            try:
                self.actor.load_weights(path + 'sac_actor.h5')
                self.critic_1.load_weights(path + 'sac_critic_1.h5')
                self.critic_2.load_weights(path + 'sac_critic_2.h5')
                print("Successfully loaded latest weights")
            except:
                print("Could not load any weights. Using initialized weights.")