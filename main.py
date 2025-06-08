import argparse
from train_lstm_embedder import train_lstm_embedder
from env import RecsysEnv
import pandas as pd
from train import train
import torch
from models import LSTMEmbedder
from dqn_agent import DQNAgent
import numpy as np
from movielens_loader import split_user_sequences
# Argument parser
parser = argparse.ArgumentParser(description="Train LSTM Embedder and RL agents")
parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes for training RL agents")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training LSTM Embedder")
parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for training LSTM Embedder")
parser.add_argument("--k", type=int, default=20, help="Number of top recommendations to evaluate")
args = parser.parse_args()
# Create a lookup dictionary for fast access

# Load dataset
ratings_path = "ml-100k/u.data"
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(ratings_path, sep='\t', names=columns)

# Prepare user sequences
user_sequences = {}
for user_id, group in df.groupby('user_id'):
    sorted_items = group.sort_values('timestamp')['item_id'].tolist()
    user_sequences[user_id] = sorted_items

# Model parameters
unique_items = df['item_id'].nunique()
seq_len = args.seq_len
batch_size = 128
embedding_dim = 64
hidden_dim = 128

# Train LSTM Embedder
model = train_lstm_embedder(
    epochs=args.epochs, seq_len=seq_len, batch_size=batch_size, 
    embedding_dim=embedding_dim, hidden_dim=hidden_dim
)
print("LSTM Embedder trained successfully!")

# Environment setup
item_count = unique_items
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 128
action_dim = item_count + 1

#lstm_embedder = LSTMEmbedder(item_count, embedding_dim, hidden_dim).to(device)
lstm_embedder = model

item_path = "ml-100k/u.item"
genre_cols = [
    "movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
item_df = pd.read_csv(item_path, sep='|', names=genre_cols, encoding='latin-1')
genre_columns = genre_cols[5:]  # Only genre columns

# Create a mapping from item_id to genre vector
item_genre_dict = {
    row['movie_id']: row[genre_columns].values.astype(np.float32)
    for _, row in item_df.iterrows()
}
# Split into train/val/test
train_sequences, val_sequences, test_sequences = split_user_sequences(user_sequences)

# ...rest of your setup (model, env, etc.)...

# Use train_sequences for training
env = RecsysEnv(train_sequences, ratings_df=df, seq_len=seq_len, item_genre_dict=item_genre_dict)

# Train RL agent
dqn_agent = train(
    agent_type='dqn',
    model=lstm_embedder,
    num_episodes=args.num_episodes,
    user_sequences=train_sequences,
    item_count=item_count,
    env=env,
    k=args.k,
    val_sequences=val_sequences  # Pass validation set to train()
)

# Train and validate DQN agent
dqn_agent = train(
    agent_type='dqn',
    model=lstm_embedder,
    num_episodes=args.num_episodes,
    user_sequences=train_sequences,
    item_count=item_count,
    env=env,
    k=args.k,
    val_sequences=val_sequences
)
# Train and validate PPO agent
ppo_agent = train(
    agent_type='ppo',
    model=lstm_embedder,
    num_episodes=args.num_episodes,
    user_sequences=train_sequences,
    item_count=item_count,
    env=env,
    k=args.k,
    val_sequences=val_sequences
)
# Train and validate REINFORCE agent
reinforce_agent = train(
    agent_type='reinforce',
    model=lstm_embedder,
    num_episodes=args.num_episodes,
    user_sequences=train_sequences,
    item_count=item_count,
    env=env,
    k=args.k,
    val_sequences=val_sequences
)
# Train and validate DDQN agent
ddqn_agent = train(
    agent_type='ddqn',
    model=lstm_embedder,
    num_episodes=args.num_episodes,
    user_sequences=train_sequences,
    item_count=item_count,
    env=env,
    k=args.k,
    val_sequences=val_sequences
)
# Train and validate A2C agent
a2c_agent = train(
    agent_type='a2c',
    model=lstm_embedder,
    num_episodes=args.num_episodes,
    user_sequences=train_sequences,
    item_count=item_count,
    env=env,
    k=args.k,
    val_sequences=val_sequences
)

# After training, evaluate all agents on the test set
from evaluate import evaluate_agent
# Create a lookup dictionary for fast access
ratings_lookup = {}
for row in df.itertuples():
    ratings_lookup[(row.user_id, row.item_id)] = row.rating

def get_rating(user_id, item_id):
    # Returns the rating if it exists, otherwise returns 0.0 (or np.nan if you prefer)
    return ratings_lookup.get((user_id, item_id), 0.0)

# Example: Compute user_stats before calling evaluate_agent
user_stats = {}
for user_id, seq in user_sequences.items():
    ratings = [get_rating(user_id, item_id) for item_id in seq]  # Implement get_rating as needed
    avg_rating = np.mean(ratings) if ratings else 0.0
    num_interactions = len(seq)
    diversity = len(set(seq)) / num_interactions if num_interactions > 0 else 0.0
    recency = 1.0  # Or your own calculation
    user_stats[user_id] = np.array([avg_rating, num_interactions / 1000, diversity, recency])
print("\n=== Test Evaluation ===")
for agent, name in [
    (dqn_agent, "DQN"),
    (ppo_agent, "PPO"),
    (reinforce_agent, "REINFORCE"),
    (ddqn_agent, "DDQN"),
    (a2c_agent, "A2C"),
]:
    test_metrics = evaluate_agent(agent, lstm_embedder, user_sequences=test_sequences, k=args.k, agent_type=name.lower(), num_users=100,user_stats=user_stats)
    print(f"{name} Test Precision@{args.k}: {test_metrics[0]}, Hit@{args.k}: {test_metrics[1]}, NDCG@{args.k}: {test_metrics[2]}")