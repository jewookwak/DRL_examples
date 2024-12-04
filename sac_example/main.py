# main.py
import os
import sys
from envs.pendulum_env import PendulumEnv
from train.config import Config
from train.trainer import SACTrainer
from test.evaluator import SACEvaluator

def main():
    # Initialize environment and configuration
    env = PendulumEnv()
    config = Config()
    
    # Create trainer
    trainer = SACTrainer(env, config)

    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # Evaluation mode
        evaluator = SACEvaluator(env, trainer, config)
        evaluation_results = evaluator.evaluate(
            num_episodes=10,  # Number of episodes to evaluate
            render=True      # Whether to render the environment
        )
        
        # Process evaluation results
        print("\nFinal Evaluation Results:")
        print(f"Mean Reward: {evaluation_results['mean_reward']:.2f}")
        if evaluation_results['mean_reward'] > config.REWARD_THRESHOLD:
            print("Goal succeeded!")
    else:
        # Training mode
        trainer.train()

if __name__ == "__main__":
    main()