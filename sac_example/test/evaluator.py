# test/evaluator.py
import tensorflow as tf
import numpy as np
import gym

class SACEvaluator:
    def __init__(self, env, trainer, config):
        self.env = env
        self.trainer = trainer
        self.config = config
        # Load the trained weights
        self.trainer.load_weights(self.config.WEIGHTS_PATH)

    def evaluate(self, num_episodes=10, render=True):
        """
        Evaluate the agent over multiple episodes
        
        Args:
            num_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render the environment
        
        Returns:
            dict: Evaluation statistics
        """
        rewards_list = []
        steps_list = []
        
        # Create render environment if needed
        render_env = None
        if render:
            try:
                render_env = gym.make("Pendulum-v0")
                render_env.reset()
            except Exception as e:
                print(f"Warning: Could not initialize rendering: {e}")
                render = False
        
        for episode in range(num_episodes):
            print(f"\nEvaluating Episode {episode + 1}/{num_episodes}")
            
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                if render and render_env:
                    try:
                        render_env.render()
                    except Exception as e:
                        render = False
                        print(f"Rendering failed: {e}")
                
                # Get deterministic action (using mean without sampling)
                mu, _ = self.trainer.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = mu.numpy()[0]
                action = np.clip(action, -self.env.action_handler.action_bound, 
                               self.env.action_handler.action_bound)
                
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
            
            rewards_list.append(total_reward)
            steps_list.append(steps)
            
            print(f"Episode {episode + 1} completed in {steps} steps with total reward {total_reward:.2f}")
        
        # Close render environment if opened
        if render_env:
            try:
                render_env.close()
            except:
                pass
        
        # Calculate statistics
        evaluation_stats = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(rewards_list),
            'std_reward': np.std(rewards_list),
            'min_reward': np.min(rewards_list),
            'max_reward': np.max(rewards_list),
            'mean_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            'rewards_list': rewards_list,
            'steps_list': steps_list
        }
        
        # Print summary statistics
        print("\nEvaluation Summary:")
        print(f"Mean Reward: {evaluation_stats['mean_reward']:.2f} ± {evaluation_stats['std_reward']:.2f}")
        print(f"Mean Steps: {evaluation_stats['mean_steps']:.2f} ± {evaluation_stats['std_steps']:.2f}")
        print(f"Min Reward: {evaluation_stats['min_reward']:.2f}")
        print(f"Max Reward: {evaluation_stats['max_reward']:.2f}")
        
        return evaluation_stats