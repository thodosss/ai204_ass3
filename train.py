import torch
from models import LSTMEmbedder
from evaluate import evaluate_agent
from dqn_agent import DQNAgent,DDQNAgent
from reinforce_agent import REINFORCEAgent
from actor_critic_agent import A2CAgent
from ppo_agent import PPOAgent
from tqdm import tqdm
import numpy as np
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(agent_type, model, num_episodes, user_sequences, item_count, env, k=10):
    embedder = model.to(device)
    
    # Update state dimension to match new state representation
    # seq_len + 4 user features + 2 temporal features
    state_dim = env.seq_len + 6
    action_dim = item_count + 1

    # Initialize metrics dictionary
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'evaluation_metrics': {
            'precision': [],
            'hit_rate': [],
            'ndcg': []
        },
        'best_reward': float('-inf'),
        'best_precision': 0.0,
        'best_hit_rate': 0.0,
        'best_ndcg': 0.0,
        'k': k  # Store k value in metrics
    }

    # Agent initialization
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif agent_type == 'reinforce':
        agent = REINFORCEAgent(state_dim, action_dim)
    elif agent_type == 'a2c':
        agent = A2CAgent(state_dim, action_dim)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim)
    elif agent_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim)
        epsilon = 1.0
        epsilon_end = 0.1
        epsilon_decay = 0.999

    for episode in tqdm(range(num_episodes), desc="Episode Progress"):
        state = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            if agent_type == 'ddqn':
                action = agent.select_action(state_tensor, epsilon)
            elif agent_type == 'ppo':
                action, log_prob, value = agent.select_action(state_tensor)
            else:
                action = agent.select_action(state_tensor.detach().cpu().numpy())

            next_item, reward, done, next_state = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)

            if agent_type == 'ddqn':
                agent.store((state_tensor, action, reward, next_state_tensor, float(done)))
                agent.update()
            elif agent_type == 'dqn':
                agent.store((state_tensor.detach().cpu().numpy(), action, reward, next_state_tensor.detach().cpu().numpy(), done))
                agent.update()
            elif agent_type == 'reinforce':
                agent.store_reward(reward)
            elif agent_type == 'a2c':
                agent.update(reward, next_state_tensor, done)
            elif agent_type == 'ppo':
                agent.store({
                    'state': state_tensor.detach(),
                    'action': action,
                    'reward': reward,
                    'log_prob': log_prob.detach(),
                    'value': value.detach(),
                    'done': float(done)
                })

            state = next_state
            total_reward += reward
            episode_length += 1

        if agent_type in ['reinforce', 'ppo']:
            agent.update()
        if agent_type == 'ddqn':
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Store episode metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['episode_lengths'].append(episode_length)
        
        # Update best reward
        if total_reward > metrics['best_reward']:
            metrics['best_reward'] = total_reward

        if episode % 100 == 0:
            print(f"Episode {episode+1} | Total Reward: {total_reward:.2f}", end="")
            if agent_type == 'ddqn':
                print(f" | Epsilon: {epsilon:.4f}", end="")
            
            # Evaluate and store metrics
            eval_metrics = evaluate_agent(agent, embedder, user_sequences, k=k, agent_type=agent_type, num_users=300)
            precision = float(eval_metrics[0].split(': ')[1])
            hit_rate = float(eval_metrics[1].split(': ')[1])
            ndcg = float(eval_metrics[2].split(': ')[1])
            
            metrics['evaluation_metrics']['precision'].append(precision)
            metrics['evaluation_metrics']['hit_rate'].append(hit_rate)
            metrics['evaluation_metrics']['ndcg'].append(ndcg)
            
            # Update best metrics
            if precision > metrics['best_precision']:
                metrics['best_precision'] = precision
            if hit_rate > metrics['best_hit_rate']:
                metrics['best_hit_rate'] = hit_rate
            if ndcg > metrics['best_ndcg']:
                metrics['best_ndcg'] = ndcg
            
            print(f" | Precision@{k}: {precision:.4f} | Hit@{k}: {hit_rate:.4f} | NDCG@{k}: {ndcg:.4f}")
    
    print(f'\nTraining complete for {agent_type} agent. ✅')
    
    # Save metrics
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f'{agent_type}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 Training metrics saved to {metrics_path}")
    
    return agent
    
