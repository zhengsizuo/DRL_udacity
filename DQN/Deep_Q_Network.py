import gym
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt
import cv2
import time


env = gym.make('Breakout-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
print('Original state shape: ', state_size)
print('Number of actions: ', env.action_space.n)

agent = Agent((32, 4, 84, 84), action_size, seed=1)  # state size (batch_size, 4 frames, img_height, img_width)
TRAIN = False  # train or test flag


def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t


def init_state(processed_obs):
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)


def dqn(n_episodes=30000, max_t=40000, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode, maximum frames
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        obs = env.reset()
        obs = pre_process(obs)
        state = init_state(obs)

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            # last three frames and current frame as the next state
            next_state = np.stack((state[1], state[2], state[3], pre_process(next_state)), axis=0)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\tEpsilon now : {:.2f}'.format(eps))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rEpisode {}\tThe length of replay buffer now: {}'.format(i_episode, len(agent.memory)))

        if np.mean(scores_window) >= 50.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint/dqn_checkpoint_solved.pth')
            break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint/dqn_checkpoint_8.pth')
    return scores


if __name__ == '__main__':
    if TRAIN:
        start_time = time.time()
        scores = dqn()
        print('COST: {} min'.format((time.time() - start_time)/60))
        print("Max score:", np.max(scores))

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint/dqn_checkpoint_8.pth'))
        rewards = []
        for i in range(10):  # episodes, play ten times
            total_reward = 0
            obs = env.reset()
            obs = pre_process(obs)
            state = init_state(obs)
            for j in range(10000):  # frames, in case stuck in one frame
                action = agent.act(state)
                env.render()
                next_state, reward, done, _ = env.step(action)
                state = np.stack((state[1], state[2], state[3], pre_process(next_state)), axis=0)
                total_reward += reward

                # time.sleep(0.01)
                if done:
                    rewards.append(total_reward)
                    break

        print("Test rewards are:", *rewards)
        print("Average reward:", np.mean(rewards))
        env.close()