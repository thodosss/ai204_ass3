import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.common(state)
        return self.actor(x), self.critic(x)

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_prob = None
        self.saved_value = None

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(device)

        probs, value = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()

        self.saved_log_prob = dist.log_prob(action)
        self.saved_value = value
        return action.item()

    def update(self, reward, next_state, done):
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        _, next_value = self.model(next_state)

        target = reward + (1 - int(done)) * self.gamma * next_value
        advantage = target - self.saved_value

        actor_loss = -self.saved_log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
