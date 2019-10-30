import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from td_plot_utils import plot_values


env = gym.make('CliffWalking-v0')

V_opt = np.zeros([4, 12])
V_opt[0] = -np.arange(3, 15)[::-1]  # [::-1]把原数列倒过来
V_opt[1] = -np.arange(3, 15)[::-1] + 1
V_opt[2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13.0
print(V_opt)

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    a_star = np.argmax(Q_s)
    policy_s[a_star] = 1 - epsilon + (epsilon / nA)
    return policy_s

def generate_episode_from_Q(env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done or len(episode)>6000:  # limit the length of episode is important!
            break
    return episode


def update_Q(target_type, episode, Q, alpha, epsilon, gamma=1.0):
    """ updates the action-value function estimate using SARSA quintuple """
    states, actions, rewards = zip(*episode)
    for i, state in enumerate(states):
        if i == (len(states)-1):
            break
        old_Q = Q[state][actions[i]]
        if target_type == 'sarsa':
            td_target = rewards[i] + gamma*Q[states[i+1]][actions[i+1]]
        elif target_type == 'q_learning':
            td_target = rewards[i] + gamma * np.max(Q[states[i + 1]])
        elif target_type == 'expected_sarsa':
            policy_s = get_probs(Q[states[i+1]], epsilon, 4)
            td_target = rewards[i] + gamma * np.dot(policy_s, Q[states[i + 1]])
        Q[state][actions[i]] = old_Q + alpha*(td_target - old_Q)

    return Q


def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        eps = 1.0 / i_episode
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = generate_episode_from_Q(env, Q, eps, nA)
        Q = update_Q('expected_sarsa', episode, Q, eps, alpha)

    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)