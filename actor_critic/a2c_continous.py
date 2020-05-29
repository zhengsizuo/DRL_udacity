"""连续动作空间下的Advantage Actor-Critic算法"""
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from tensorboardX import SummaryWriter
from actor_critic.ac_model import Actor, Critic

ACTOR_LR = 1e-3
CRITIC_LR = 1e-2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_BETA = 0.01
EPISODES = 2000  # 收集3000条序列
MAX_STEP = 200  # 每条序列最多200步
USE_GAE = True

env = gym.make('Pendulum-v0')
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

def calc_logprob(mu_v, logstd_v, actions_v):
    # 计算高斯分布的log probability
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * np.pi * torch.exp(logstd_v)))
    return p1 + p2


def generalized_advantage_estimation(memories):
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


ac_losses =[]
cr_losses = []
def a2c(n_episodes, max_step, gamma):
    """增加了baseline和策略的熵的PG算法"""
    total_rewards = []  # 保存每一个序列的回报

    for i_episode in range(n_episodes):
        state = env.reset()
        memories = []

        for t in range(max_step):
            action = actor.choose_action(state, device)
            state_v = torch.from_numpy(state).float().to(device)
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

        # 对reward进行normalize
        rewards = [m.r for m in memories]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)

        adv_funcs = []
        # 对该条序列进行训练
        actor_loss = torch.tensor([0.])
        critic_loss = torch.tensor([0.])
        if USE_GAE:
            memories = generalized_advantage_estimation(memories)

        for t in reversed(range(len(memories)-1)):
            m = memories[t]
            # m_r = (m.r - mean_r) / std_r
            # adv = m_r + gamma*memories[t+1].value - m.value
            adv = m.adv
            critic_loss += adv ** 2
            adv_funcs.append(adv.item())

            state_v = torch.from_numpy(m.s).float().to(device)
            mu_v = actor.forward(state_v)
            action_v = torch.tensor(m.a)
            log_prob = calc_logprob(mu_v, action_v, actor.logstd)
            actor_loss += -log_prob * adv.detach()


        # adv_funcs = (adv_funcs - np.mean(adv_funcs)) / np.std(adv_funcs)
        # adv_funcs = adv_funcs[::-1]
        # for t in reversed(range(len(memories)-1)):
        #     m = memories[t]
        #     state_v = torch.from_numpy(m.s).float().to(device)
        #     mu_v = actor.forward(state_v)
        #     action_v = torch.tensor(m.a)
        #     log_prob = calc_logprob(mu_v, action_v, actor.logstd)
        #     actor_loss += -log_prob * adv_funcs[t]
        # print("Advantage functions:", adv_funcs)

        critic_optimizer.zero_grad()
        critic_loss = critic_loss/len(memories)
        critic_loss.backward()
        critic_optimizer.step()
        #cr_losses.append(critic_loss.item())

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # print(actor.logstd)
        #ac_losses.append(actor_loss.item())
        total_rewards.append(sum(rewards))

        # writer.add_scalar("baseline", baseline, i_episode)
        # writer.add_scalar("policy_loss", policy_loss.item(), i_episode)
        # writer.add_scalar("entropy_loss", entropy_loss.item(), i_episode)
        # writer.add_scalar("total_loss", total_loss.item(), i_episode)
        recent_reward = np.mean(total_rewards[-100:])
        if i_episode % 10 == 0:
            cr_losses.append(critic_loss.item())
            ac_losses.append(actor_loss.item())
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, recent_reward))
        if recent_reward >= -200.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, recent_reward))
            break

    # writer.close()
    # torch.save(policy.state_dict(), 'model/pg_checkpoint.pth')
    return total_rewards


scores = a2c(EPISODES, MAX_STEP, GAMMA)
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