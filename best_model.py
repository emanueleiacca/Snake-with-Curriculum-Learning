import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from snakeenv2 import SnakeEnv

# Load the best model.
model = PPO.load("Models_trained\model_S_8_7_curr.zip")
env = SnakeEnv(width=25, height=25)
# Evaluate the agent's performance on a set of test episodes.
total_reward = 0
for i in range(100):
    obs = env.reset()  
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

# Calculate the mean score.
mean_score = total_reward / 100

# Print the mean score.
print(f"Average score: {mean_score}")
