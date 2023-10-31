import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from snakeenv import SnakeEnv
import sys
sys.setrecursionlimit(10000)

# Define your curriculum schedule
curriculum_schedule = [4, 5, 6, 7, 8, 9, 10, 15, 20]  # Board sizes: 4x4, 5x5, 6x6, 7x7, 8x8, 9x9, 10x10, 15x15, 20x20
# Define your curriculum schedule
curriculum_schedule = [(size, curriculum) for size, curriculum in enumerate(curriculum_schedule)]
curriculum_levels = len(curriculum_schedule)
total_timesteps = 500000  # Total training timesteps per curriculum level
# Define your early stopping criterion
plateau_threshold = 50

def early_stopping_criterion(scores):
    # Check if the scores list is empty.
    if not scores:
        return False
    if len(scores) == 0:
        return False

    # Calculate the mean score over the past few levels of the curriculum.
    recent_scores = scores[-plateau_threshold:]
    recent_mean_score = np.mean(recent_scores)

    # If the mean score has not improved for the past few levels, stop training.
    if recent_mean_score <= max(scores[:-plateau_threshold]):
        return True

    return False
def print_training_results(scores, mean_score, highest_reward, highest_single):
    print("----------------------------------------------------")
    print(f"Scores: {scores}")
    print(f"Mean score: {mean_score}")
    print(f"Highest reward: {highest_reward}")
    print(f"Highest single score: {highest_single}")
    print("----------------------------------------------------")

# Define your training loop
for level, (size, curriculum) in enumerate(curriculum_schedule):
    env = SnakeEnv(width=size, height=size, curriculum=curriculum)  # Initialize the SnakeEnv with the current board size and curriculum
    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose=1)

    highest_reward = -100
    highest_single = -100
    i = 1
    plateau = 0

    while i * 10000 <= total_timesteps:
        model.learn(10000, reset_num_timesteps=False, log_interval=1000)

        print(f"S={size}, Curr={curriculum}: Testing agent after {i * 10000} iterations...")
        scores = []

        for _ in range(10):
            obs = env.reset()  # Reset the environment
            one = False
            total_reward = 0

            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            scores.append(total_reward)

        # Calculate the mean score.
        if scores:
            mean_score = np.mean(scores)
        else:
            mean_score = 0
        # Save the model if the agent achieves a new high score.
        if mean_score > highest_reward:
            model.save(f"model_S_{size}_{i}_curr")
            highest_reward = mean_score
        print_training_results(scores, mean_score, highest_reward, highest_single)
        if early_stopping_criterion(scores):
            break

        i += 1