import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, lr=1e-3, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = gamma
        self.epsilon = 0.1
        self.update_steps = 0
        self.sync_interval = 100  # update target network every N steps

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_net[-1].out_features - 1)
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
            return self.q_net(state).argmax().item()

    def store(self, transition):
        """transition: tuple (state, action, reward, next_state, done)"""
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.functional.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_steps += 1
        if self.update_steps % self.sync_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=10000, tau=0.01):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32).to(device)
                q_values = self.q_net(state)
            return q_values.argmax().item()

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in next_states]).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Double DQN update
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_q_net(next_states).gather(1, next_actions)
        target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


