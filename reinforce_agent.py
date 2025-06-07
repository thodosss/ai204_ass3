import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.model(state)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        G = 0
        loss = 0
        returns = []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).float().to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.rewards = []
