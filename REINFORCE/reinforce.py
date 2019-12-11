import gym
# gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

env = gym.make('CartPole-v0')
# env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN = False  # train or test flag


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)  # 离散型概率分布
        action = m.sample()
        return action.item(), m.log_prob(action)


policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def reinforce(n_episodes=10000, max_t=1000, gamma=1, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    torch.save(policy.state_dict(), 'pg_checkpoint.pth')
    return scores


if __name__ == '__main__':
    if TRAIN:
        start_time = time.time()
        scores = reinforce()
        print('COST: {} min'.format((time.time() - start_time)/60))
        print("Max score:", np.max(scores))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        # load the weights from file
        policy.load_state_dict(torch.load('pg_checkpoint.pth'))
        rewards = []
        for i in range(10):  # episodes, play ten times
            total_reward = 0
            state = env.reset()
            for j in range(1000):  # frames, in case stuck in one frame
                action, _ = policy.act(state)
                env.render()
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward

                # time.sleep(0.01)
                if done:
                    rewards.append(total_reward)
                    break

        print("Test rewards are:", *rewards)
        print("Average reward:", np.mean(rewards))
        env.close()
