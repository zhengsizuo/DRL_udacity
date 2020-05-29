"""基于A2C的PPO算法，采用clip的目标函数"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from tensorboardX import SummaryWriter
from PPO.ac_model import Actor, Critic

# 超参数设置
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_BETA = 0.01

EPISODES = 3000  # 收集3000条序列
MAX_STEP = 193  # 每条序列最多200步
PPO_EPOCH = 5
CLIP_EPS = 0.2
BATCH_SIZE = 32

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
print("env.action_space: ", env.action_space.shape[0])
print("env.observation_space: ", env.observation_space.shape[0])
obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
actor = Actor(obs_size, act_size).to(device)
critic = Critic(obs_size).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

Memory = namedtuple('Memory', ['s', 'a', 'r', 's_', 'done', 'value', 'adv'])


def generalized_advantage_estimation(memories):
    """计算每个time_step对应的泛化优势函数估计，返回新的memories"""
    new_memories = []
    gae = 0
    # 对reward进行normalize
    rewards = [m.r for m in memories]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    for t in reversed(range(len(memories)-1)):
        m = memories[t]
        m_r = (m.r - mean_r) / std_r
        if m.done:
            gae = m_r
        else:
            td_error = m_r + GAMMA*memories[t+1].value - m.value
            gae = gae + GAMMA*GAE_LAMBDA*td_error

        new_memories.insert(0, Memory(s=m.s, a=m.a, s_=m.s_, r=gae+m.value, done=m.done, value=m.value, adv=gae))

    return new_memories


def old_log_policy_prob(batch, actor, device):
    states_batch = np.array([b.s for b in batch])
    actions_batch = np.array([b.a for b in batch])
    state_v = torch.FloatTensor(states_batch).to(device)
    action_v = torch.FloatTensor(actions_batch).to(device)
    gaussian_dist = actor(state_v)

    return gaussian_dist.log_prob(action_v)


ac_losses =[]
cr_losses = []
def ppo(n_episodes, max_step, gamma):
    total_rewards = []  # 保存每一个序列的回报

    for i_episode in range(n_episodes):
        state = env.reset()
        memories = []

        # 收集一条序列的信息
        for t in range(max_step):
            state_v = torch.from_numpy(state).float().to(device)
            dist = actor(state_v)
            action = dist.sample().cpu().numpy()
            action = np.clip(action, -2, 2)
            value_est = critic(state_v)

            next_state, reward, done, _ = env.step(action)
            if done:
                memories.append(
                    Memory(s=state, a=action, s_=next_state, r=0, done=done, value=value_est, adv=0))
                break
            else:
                memories.append(
                Memory(s=state, a=action, s_=next_state, r=reward+8/8, done=done, value=value_est, adv=0))

            # 下一个time step
            state = next_state

        rewards = sum([m.r for m in memories])
        batch = generalized_advantage_estimation(memories)
        batch_adv = torch.FloatTensor([b.adv for b in batch])
        batch_adv = (batch_adv - torch.mean(batch_adv)) / torch.std(batch_adv)
        # Compute the policy probability with the old policy network
        old_log_policy = old_log_policy_prob(batch, actor, device)  # torch.size([192. 1])

        for _ in range(PPO_EPOCH):
            # compute the loss and optimize over mini batches of size BATCH_SIZE
            for mb in range(0, len(batch), BATCH_SIZE):
                mini_batch = batch[mb:mb + BATCH_SIZE]
                minib_old_log_policy = old_log_policy[mb:mb + BATCH_SIZE]  # torch.size([32, 1])
                minib_adv = batch_adv[mb:mb + BATCH_SIZE]
                minib_action = torch.FloatTensor([m.a for m in mini_batch])

                minib_returns = torch.FloatTensor([m.r for m in mini_batch])
                minib_states = torch.FloatTensor([m.s for m in mini_batch])
                minib_values = critic(minib_states)  # torch.size([32, 1])

                # 训练critic
                value_loss = F.mse_loss(minib_values.squeeze(-1), minib_returns)
                critic_optimizer.zero_grad()
                value_loss.backward()
                critic_optimizer.step()

                # 训练actor
                minib_dist = actor(minib_states)
                new_log_policy = minib_dist.log_prob(minib_action)
                rt_theta = (new_log_policy - minib_old_log_policy.detach()).exp()
                # print(rt_theta.size())
                minib_adv = minib_adv.unsqueeze(-1)
                surr1 = rt_theta * minib_adv
                surr2 = minib_adv * torch.clamp(rt_theta, 1-CLIP_EPS, 1+CLIP_EPS)
                actor_loss = -torch.min(surr1, surr2).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                # print(actor.logstd)

        total_rewards.append(rewards)
        recent_reward = np.mean(total_rewards[-100:])
        if i_episode % 100 == 0:
            ac_losses.append(actor_loss.item())
            cr_losses.append(value_loss.item())
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, recent_reward))
        if recent_reward >= -200.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, recent_reward))
            break

    return total_rewards



scores = ppo(EPISODES, MAX_STEP, GAMMA)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

plt.figure(2)
plt.title("actor loss")
plt.plot(ac_losses)
plt.figure(3)
plt.title("critic loss")
plt.plot(cr_losses)

plt.show()