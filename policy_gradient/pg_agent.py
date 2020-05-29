"""搭建策略网络的结构"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Policy(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # 对每一行进行softmax


def choose_act(policy, state, device=torch.device("cpu")):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy.forward(state).cpu()
    m = Categorical(probs)
    action = m.sample()
    # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
    return action.item(), m.log_prob(action), probs

# np.random.seed(0)
# # 测试模型维度
pg_model = Policy(state_size=4, hidden_size=16, action_size=2)
# # # state = torch.randn(1, 4)
# # # action_prob = pg_model.forward(state)
# # # print(state)
# # # print(action_prob)
# # #
state = np.random.randn(4)
action_ix, log_prob, prob = choose_act(pg_model, state)
print(action_ix, log_prob, prob)
entropy = prob * torch.log(prob)
print(torch.log(prob))
print(-entropy.sum(dim=1))
