import argparse
from train_lstm_embedder import train_lstm_embedder
from env import RecsysEnv
import pandas as pd
from train import train
import torch
from models import LSTMEmbedder
from dqn_agent import DQNAgent

# Argument parser
parser = argparse.ArgumentParser(description="Train LSTM Embedder and RL agents")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for training RL agents")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training LSTM Embedder")
parser.add_argument("--seq_len", type=int, default=5, help="Sequence length for training LSTM Embedder")
parser.add_argument("--k", type=int, default=10, help="Number of top recommendations to evaluate")
args = parser.parse_args()

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
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
lstm_embedder = LSTMEmbedder(item_count, embedding_dim, hidden_dim).to(device)
lstm_embedder = model
env = RecsysEnv(user_sequences, ratings_df=df, seq_len=seq_len)

# Train RL agents
#dqn_agent = train(agent_type='dqn', model=lstm_embedder, num_episodes=args.num_episodes, user_sequences=user_sequences, item_count=item_count, env=env, k=args.k)
#reinforce_agent = train(agent_type='reinforce', model=lstm_embedder, num_episodes=args.num_episodes, user_sequences=user_sequences, item_count=item_count, env=env, k=args.k)
#a2c_agent = train(agent_type='a2c', model=lstm_embedder, num_episodes=args.num_episodes, user_sequences=user_sequences, item_count=item_count, env=env, k=args.k)
ppo_agent = train(agent_type='ppo', model=lstm_embedder, num_episodes=args.num_episodes, user_sequences=user_sequences, item_count=item_count, env=env, k=args.k)
ddqn_agent = train(agent_type='ddqn', model=lstm_embedder, num_episodes=args.num_episodes, user_sequences=user_sequences, item_count=item_count, env=env, k=args.k)
