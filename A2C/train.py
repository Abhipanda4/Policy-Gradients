import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import gym
import numpy as np
import matplotlib.pyplot as plt

from model import ActorNetwork, CriticNetwork
from torchviz import make_dot

actor = ActorNetwork(4, 2)
critic = CriticNetwork(4)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=8e-4)
env = gym.make('CartPole-v0')
GAMMA = 0.99
N_EPISODES = 20000
LOG_STEPS = 100
SAVE_STEPS = 100

def select_action(S):
    '''
    select action based on currentr state
    args:
        S: current state
    returns:
        action to take, log probability of the chosen action
    '''
    action_probs = actor(S)

    m = Categorical(action_probs)
    A = m.sample()
    L = m.log_prob(A)
    return A.item(), L

def train_one_episode():
    is_done = False
    S = env.reset()

    rewards = []
    state_vals = []
    log_probs = []

    while not is_done:
        S = Variable(torch.FloatTensor(S).unsqueeze(0))
        state_vals.append(critic(S))
        A, L = select_action(S)
        S, R, is_done, _ = env.step(A)
        if is_done:
            R = -5
        rewards.append(R)
        log_probs.append(L)

    R = 0
    actor_loss = 0
    critic_loss = 0
    for i in reversed(range(len(rewards))):
        R = R + rewards[i] * GAMMA
        advantage = R - state_vals[i]

        actor_loss += -advantage * log_probs[i]
        critic_loss += advantage ** 2

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    critic_loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()

    episode_reward = len(rewards)
    return episode_reward



def main():
    running_reward = 0
    rewards = []
    episodes = []
    for i in range(N_EPISODES):
        R = train_one_episode()

        # track statistics
        if running_reward != 0:
            running_reward = 0.9 * running_reward + 0.1 * R
        else:
            running_reward = R

        # save statistics
        if i % SAVE_STEPS == 0:
            rewards.append(running_reward)
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
