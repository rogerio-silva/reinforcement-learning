import os
import gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

log_path = os.path.join('Training', 'Logs')

env_name = 'CartPole-v0'

env = gym.make(env_name)
env = DummyVecEnv([lambda:  env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
#
# episodes = 20
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()
