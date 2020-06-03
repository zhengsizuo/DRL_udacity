"""搭建不共享网络的Actor-Critic结构"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

HID_SIZE = 128


class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Actor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            # nn.Linear(HID_SIZE, HID_SIZE),
            # nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))  # 独立的参数

    def forward(self, x):
        mu = self.mu(x)
        std = self.logstd.exp().expand_as(mu)  # 扩充为跟mu的大小一样
        m = Normal(mu, std)
        return m


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


def test_net(net, env, action_bound=1, count=10, device="cpu", render_flag=True):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            if render_flag:
                env.render()
            state = torch.from_numpy(obs).float().to(device)
            dist = net(state)
            action = dist.sample().cpu().numpy()
            action = np.clip(action, -action_bound, action_bound)
            obs, reward, done, _ = env.step(action)

            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


# 测试网络的参数size是否设置正确，以及随机的
# import gym
# env = gym.make('Pendulum-v0')
# print("env.action_space: ", env.action_space.shape[0])
# print("env.observation_space: ", env.observation_space.shape[0])
# obs_size = env.observation_space.shape[0]
# act_size = env.action_space.shape[0]
#
# actor = Actor(obs_size, act_size)
# avg_r, avg_step = test_net(actor, env, action_bound=1, count=100, device="cpu", render_flag=False)
# print(avg_r, avg_step)
