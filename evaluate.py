import torch
import numpy as np

def hit_at_k(recommended_items, true_item, k):
    return int(true_item in recommended_items[:k])

def precision_at_k(recommended_items, true_item, k):
    return int(true_item in recommended_items[:k]) / k

def ndcg_at_k(recommended_items, true_item, k):
    if true_item in recommended_items[:k]:
        rank = recommended_items[:k].index(true_item)
        return 1.0 / np.log2(rank + 2)  # +2 because rank is 0-based and log2(1) = 0
    return 0.0

def evaluate_agent(agent, embedder, user_sequences, k, num_users, agent_type):
    precision_scores, hit_scores, ndcg_scores = [], [], []
    device = next(embedder.parameters()).device

    # Get state dimension based on agent type
    if agent_type == 'dqn':
        state_dim = agent.q_net[0].in_features
    elif agent_type == 'ddqn':
        state_dim = agent.q_net.net[0].in_features
    elif agent_type == 'reinforce':
        state_dim = agent.policy.model[0].in_features
    elif agent_type == 'a2c':
        state_dim = agent.model.common[0].in_features
    elif agent_type == 'ppo':
        state_dim = agent.policy.shared[0].in_features
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    seq_len = state_dim - 6  # Total features - (user features + temporal features)

    sampled_users = list(user_sequences.keys())[:num_users]
    for user in sampled_users:
        seq = user_sequences[user]
        if len(seq) <= seq_len + 1:
            continue
        for i in range(seq_len, len(seq) - 1):
            # Create state features
            seq_features = np.zeros(seq_len)
            seq_features[-seq_len:] = seq[i-seq_len:i]
            
            # Create dummy user features (since we don't have user stats in evaluation)
            user_features = np.array([0.5, 0.5, 0.5, 0.5])  # Default values
            
            # Create temporal features
            temporal_features = np.array([
                i / len(seq),  # Progress
                1.0  # History fullness
            ])
            
            # Combine features
            state = np.concatenate([seq_features, user_features, temporal_features])
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            with torch.no_grad():
                if agent_type == 'dqn':
                    logits = agent.q_net(state_tensor)
                elif agent_type == 'ddqn':
                    logits = agent.q_net(state_tensor)
                elif agent_type == 'reinforce':
                    probs = agent.policy(state_tensor)
                    logits = probs
                elif agent_type == 'a2c':
                    probs, _ = agent.model(state_tensor)
                    logits = probs
                elif agent_type == 'ppo':
                    probs, _ = agent.policy(state_tensor)
                    logits = probs
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                # Get top k predictions
                if isinstance(logits, torch.Tensor):
                    if len(logits.shape) == 1:
                        logits = logits.unsqueeze(0)  # Add batch dimension if missing
                    top_k_preds = torch.topk(logits, k).indices[0].cpu().numpy().tolist()
                else:
                    # Handle non-tensor outputs (e.g., from REINFORCE)
                    top_k_preds = np.argsort(logits)[-k:][::-1].tolist()

                ground_truth = seq[i]

                precision_scores.append(precision_at_k(top_k_preds, ground_truth, k))
                hit_scores.append(hit_at_k(top_k_preds, ground_truth, k))
                ndcg_scores.append(ndcg_at_k(top_k_preds, ground_truth, k))

    return [
        f"Precision@{k}: {np.mean(precision_scores):.5f}",
        f"Hit@{k}: {np.mean(hit_scores):.5f}",
        f"NDCG@{k}: {np.mean(ndcg_scores):.5f}"
    ]
