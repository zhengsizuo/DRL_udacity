"""策略梯度算法"""
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from actor_critic.ac_model import Actor

ENV_NAME = 'Pendulum-v0'
LEARNING_RATE = 0.01
GAMMA = 0.99
ENTROPY_BETA = 0.01
EPISODES = 2000  # 收集3000条序列
MAX_STEP = 1000  # 每条序列最多1000步

env = gym.make(ENV_NAME)
# writer = SummaryWriter(comment="-cartpole-pg")
# 策略梯度算法方差很大，设置seed以保证复现性
#env.seed(1)
#torch.manual_seed(0)
print("env.action_space: ", env.action_space.shape[0])
print("env.observation_space: ", env.observation_space.shape[0])
obs_size = env.observation_space.shape[0]
act_size = env.action_space.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = Actor(obs_size, act_size).to(device)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

def norm_reward(R_tau):
    for i in range(1, len(R_tau)):
        R_tau[i] = R_tau[i-1] + R_tau[i]

    return (R_tau - np.mean(R_tau)) / np.std(R_tau) + + 1e-7


def Vannilla_PG(n_episodes, max_step, gamma):
    total_rewards = []  # 保存每一个序列的回报
    for i_episode in range(n_episodes):
        state = env.reset()
        ep_rewards = []  # 保存当前序列每一步的回报
        saved_log_probs = []  # 保存每一步动作的log probability
        for t in range(max_step):
            action, log_prob = policy.choose_action(state, device)
            action = np.clip(action, -1, 1)
            action = np.array([action])
            next_state, reward, done, _ = env.step(action)
            state = next_state
            saved_log_probs.append(log_prob)
            ep_rewards.append(reward/10)
            if done:
                break

        total_rewards.append(sum(ep_rewards))
        discounts = [gamma**i for i in range(len(ep_rewards))]
        R_tau = [a*b for a, b in zip(discounts[::-1], ep_rewards)]
        R_tau = norm_reward(R_tau)
        # baseline = np.mean(total_rewards)  # 过去所有序列的回报均值作为baseline
        # R_tau = R_tau - baseline
        # print(R_tau)
        # 对该条序列进行训练
        policy_loss = torch.tensor([0.])
        for i, log_pi in enumerate(saved_log_probs):
            policy_loss += -log_pi * R_tau[i]
        # print("Policy loss: ", policy_loss.item())

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        recent_reward = np.mean(total_rewards[-100:])
        if i_episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, recent_reward))
        if recent_reward >= -10.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, recent_reward))
            break

    torch.save(policy.state_dict(), 'model/pg_continous.pth')
    return total_rewards



# def watch_agent(model_flie='model/pg_checkpoint.pth'):
#     # load the weights from file
#     policy.load_state_dict(torch.load(model_flie))
#     rewards = []
#     for i in range(10):  # episodes, play ten times
#         total_reward = 0
#         state = env.reset()
#         for j in range(1000):  # frames, in case stuck in one frame
#             action, _, _ = choose_act(policy, state)
#             env.render()
#             next_state, reward, done, _ = env.step(action)
#             state = next_state
#             total_reward += reward
#
#             if done:
#                 rewards.append(total_reward)
#                 break
#
#     print("Test rewards are:", *rewards)
#     print("Average reward:", np.mean(rewards))
#     env.close()


scores = Vannilla_PG(EPISODES, MAX_STEP, GAMMA)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# watch_agent()