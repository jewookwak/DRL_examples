# main.py
import os
import sys
from envs.pendulum_env import PendulumEnv
from train.config import Config
from train.trainer import DDPGTrainer
from test.evaluator import DDPGEvaluator
# import numpy as np
# from datetime import datetime
# import tensorflow as tf

# # TensorBoard setup
# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = os.path.join('logs', 'DDPG', current_time)
# summary_writer = tf.summary.create_file_writer(train_log_dir)
# best_reward = float('-inf')
# save_interval =20
# def train():
#     # Create save directories if they don't exist
#     os.makedirs("./save_weights", exist_ok=True)

#     # Initialize environment, configuration and trainer
#     env = PendulumEnv()
#     config = Config()
#     trainer = DDPGTrainer(env, config)
    
#     score_avg = 0  # score_avg 초기화
    
#     # Training loop
#     print("Starting training...")
#     for e in range(config.MAX_EPISODES):
#         state = env.reset()
#         episode_reward = 0
#         done = False
#         time = 0
#         while not done:
#             action = trainer._get_action(state, trainer.prev_noise)
#             next_state, reward, done, _ = env.step(action)
            
#             # Save the experience to the replay buffer
#             trainer.buffer.add_buffer(state, action, reward, next_state, done)
            
#             if trainer.buffer.buffer_count() > config.MIN_BUFFER_SIZE:
#                 trainer._train_step()
            
#             state = next_state
#             episode_reward += reward
#             time += 1
        
#         # Update target networks and save the model periodically
#         trainer.update_target_network(config.TAU)
#         trainer._save_weights()
        
#         # Calculate average score
#         score_avg = 0.9 * score_avg + 0.1 * episode_reward if score_avg != 0 else episode_reward
        
#         print(f"Episode: {e+1}/{config.MAX_EPISODES} | Score: {episode_reward:.2f} | Score Average: {score_avg:.2f}")
#         # TensorBoard에 reward와 average reward 기록
#         with summary_writer.as_default():
#             tf.summary.scalar('reward/episode_reward', episode_reward, step=e)
#             tf.summary.scalar('reward/average_reward', score_avg, step=e)
#             summary_writer.flush()  # 강제 플러시
        
#         # Early stopping if the performance is good enough
#         if score_avg > 180:
#             print("Training finished successfully!")
#             trainer._save_weights()
#             break
#         elif episode_reward > best_reward:
#             trainer._save_weights(best=True)
#         elif e % save_interval == 0:  # 주기적 저장
#             trainer._save_weights(best=False)

# def evaluation():
#     print("\nStarting evaluation mode...")
#     env = PendulumEnv()
#     config = Config()
#     trainer = DDPGTrainer(env, config)
#     evaluator = DDPGEvaluator(env, trainer, config)
    
#     print("\nRunning 10 evaluation episodes...")
#     total_reward = 0
#     rewards = []
    
#     for i in range(10):
#         try:
#             episode_reward = evaluator.evaluate(render=True)
#         except Exception as e:
#             print(f"Error during evaluation: {e}")
#             print("Trying without rendering...")
#             episode_reward = evaluator.evaluate(render=False)
        
#         rewards.append(episode_reward)
#         total_reward += episode_reward
#         print(f"Test Episode {i+1}/10 | Reward: {episode_reward:.2f}")
    
#     avg_reward = total_reward / 10
#     print(f"\nEvaluation Results:")
#     print(f"Average Reward: {avg_reward:.2f}")
#     print(f"Best Episode Reward: {max(rewards):.2f}")
#     print(f"Worst Episode Reward: {min(rewards):.2f}")
#     print(f"Reward Standard Deviation: {np.std(rewards):.2f}")

def main():
    # Initialize environment and configuration
    env = PendulumEnv()
    config = Config()
    
    # Create trainer
    trainer = DDPGTrainer(env, config)
    if len(sys.argv) > 1 and sys.argv[1] == 'eval': # Evaluate trained agent        
        evaluator = DDPGEvaluator(env, trainer, config)
        # 10개 에피소드 평가 (렌더링 활성화)
        evaluation_results = evaluator.evaluate(
            num_episodes=10,  # 평가할 에피소드 수 
            render=True       # 환경 렌더링 여부
        )
        
        # 평가 결과 분석
        print("mean reward:", evaluation_results['mean_reward']) 
        # 필요한 경우 추가 처리 가능
        if evaluation_results['mean_reward'] > 180:
            print("Goal succeeded!")
    else:# Train agent
        trainer.train()

# def main():
#     if len(sys.argv) > 1 and sys.argv[1] == 'eval':
#         evaluation()
#     else:
#         train()

if __name__ == "__main__":
    main()