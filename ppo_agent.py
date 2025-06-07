import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, clip_eps=0.2):
        self.policy = PPOActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = clip_eps  
        self.memory = []


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        probs, value = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        # Extract data
        states = torch.stack([m['state'] for m in self.memory]).detach()
        actions = torch.tensor([m['action'] for m in self.memory]).to(device)
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory]).detach()
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]

        # Compute returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(4):  # K epochs
            probs, values = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Advantage estimation
            advantages = returns - values.squeeze().detach()

            # Ratio of new/old policy
            ratios = torch.exp(log_probs - old_log_probs)

            # PPO clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []
