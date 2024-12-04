# main.py
import os
import sys
import argparse
import gym
from train.config import Config
from train.trainer import DQNTrainer
from test.evaluator import DQNEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='DQN for CartPole')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--render', action='store_true', help='Render environment during evaluation')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # render 옵션에 따라 환경 생성 방식을 다르게 설정
    if args.eval and args.render:
        env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=200)
    else:
        env = gym.make('CartPole-v1', max_episode_steps=200)
    
    config = Config()
    trainer = DQNTrainer(env, config)
    
    if args.eval:
        print("Starting evaluation...")
        evaluator = DQNEvaluator(env, trainer, config)
        evaluation_results = evaluator.evaluate(
            num_episodes=args.episodes,
            render=args.render
        )
        
        print("\nFinal Evaluation Results:")
        print(f"Mean Reward: {evaluation_results['mean_reward']:.2f}")
        if evaluation_results['mean_reward'] > config.REWARD_THRESHOLD:
            print("Goal succeeded!")
    else:
        print("Starting training...")
        trainer.train()

if __name__ == "__main__":
    main()