"""Test environments of gym"""
import gym
import torch
import numpy as np

env = gym.make('Pendulum-v0')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space)
print("env.observation_space.high: ", env.observation_space.high)
print("env.observation_space.low: ", env.observation_space.low)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        state = torch.from_numpy(observation).float().unsqueeze(0)
        # print(state.size())
        action = env.action_space.sample()
        print(np.shape(action))
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()