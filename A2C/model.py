import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, inp_dim, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 24)
        self.fc2 = nn.Linear(24, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=1)
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, inp_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_val = F.relu(self.fc2(x))
        return state_val
