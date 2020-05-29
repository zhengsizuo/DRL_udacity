"""搭建不共享网络的Actor-Critic结构"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

HID_SIZE = 64


class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_size, HID_SIZE)
        self.fc2 = nn.Linear(HID_SIZE, act_size)
        self.logstd = nn.Parameter(torch.zeros(act_size))  # 怎么对该参数进行训练

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)

        return torch.tanh(x)

    def choose_action(self, state, device):
        state = torch.from_numpy(state).float().to(device)
        mu = self.forward(state)
        mu = mu.data.cpu().numpy()
        logstd = self.logstd.data.cpu().numpy()
        actions = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
        action = np.clip(actions, -2, 2)

        return action


class Critic(nn.Module):
    def __init__(self, obs_size):
        super(Critic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            # nn.Linear(HID_SIZE, HID_SIZE),
            # nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


def test_net(net, env, count=100, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            # print("step: ", steps)
            action = net.choose_action(obs, device)
            obs, reward, done, _ = env.step(action)

            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


# import gym
# env = gym.make('Pendulum-v0')
# print("env.action_space: ", env.action_space.shape[0])
# print("env.observation_space: ", env.observation_space.shape[0])
# obs_size = env.observation_space.shape[0]
# act_size = env.action_space.shape[0]
#
# actor = Actor(obs_size, act_size)
# avg_r, avg_step = test_net(actor, env)
# print(avg_r, avg_step)
