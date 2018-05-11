import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable

import gym
import numpy as np
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# global parameters
model = Policy(4, 2)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
env = gym.make('CartPole-v0')
N_EPISODES = 20000
GAMMA = 0.99
LOG_STEPS = 100
SAVE_STEPS = 50

def select_action(S):
    '''
    select action based on currentr state
    args:
        S: current state
    returns:
        action to take, log probability of the chosen action
    '''
    S = torch.from_numpy(S).float().unsqueeze(0)
    S = Variable(S)
    out = model(S)

    m = Categorical(out)
    A = m.sample()
    L = m.log_prob(A)
    return A.item(), L


def run_one_episode():
    '''
    Simulate one full episode
    returns:
        a list of rewards at each step,
        a list of log probs of actions taken at each step
    '''
    rewards = []
    log_probs = []
    is_done = False
    S = env.reset()
    while not is_done:
        A, log_A = select_action(S)
        S, R, is_done, _ = env.step(A)
        rewards.append(R)
        log_probs.append(log_A)
    rewards[-1] = -1
    return rewards, log_probs

def improve_policy(R, L):
    '''
    perform weight updates using REINFORCE algo
    args:
        R: list of rewards at each step of episode
        L: list of log prob of each action chosen
    returns:
        loss in the episode
        total reward for this episode
    '''
    discounted_R = []
    total_R = 0
    # compute discounted return from each action
    for r in reversed(R):
        total_R = r + GAMMA * total_R
        discounted_R.insert(0, total_R)

    # for normalization
    mean_R = np.mean(discounted_R)
    stddev_R = np.std(discounted_R)

    # compute normalized advantage
    discounted_R = (discounted_R - mean_R) / (stddev_R + 0.0000001)
    discounted_R = torch.from_numpy(discounted_R)

    policy_loss = []
    for r, l in zip(discounted_R, L):
        policy_loss.append(-1 * l * r)

    # Gradient Descent
    optimizer.zero_grad()
    loss = torch.cat(policy_loss).sum()
    loss.backward()
    optimizer.step()
    return loss.item(), len(R)

def main():
    running_reward = 0
    rewards = []
    losses = []
    episodes = []
    for i in range(N_EPISODES):
        R, L = run_one_episode()
        loss, R = improve_policy(R, L)

        # track statistics
        if running_reward != 0:
            running_reward = 0.9 * running_reward + 0.1 * R
        else:
            running_reward = R

        # save statistics
        if i % SAVE_STEPS == 0:
            rewards.append(running_reward)
            losses.append(loss)
            episodes.append(i)

        # log info
        if (i + 1) % LOG_STEPS == 0:
            print("Completed [%6d] simulations... Running reward: [%.5f]" %(i + 1, running_reward))

    # plot rewards
    plt.plot(episodes, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Running Reward")
    plt.show()

if __name__ == "__main__":
    main()
