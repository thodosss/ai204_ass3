import pandas as pd
from collections import defaultdict
from datasets import load_dataset
def split_user_sequences(user_sequences, val_ratio=0.1, test_ratio=0.1):
    train_seqs, val_seqs, test_seqs = {}, {}, {}
    for user, seq in user_sequences.items():
        n = len(seq)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        train_seqs[user] = seq[:n - n_val - n_test]
        val_seqs[user] = seq[n - n_val - n_test : n - n_test]
        test_seqs[user] = seq[n - n_test:]
    return train_seqs, val_seqs, test_seqs

def create_user_sequences(df, min_len=6):
    user_sequences = defaultdict(list)
    df = df.sort_values(by=['user_id', 'timestamp'])  # sort chronologically
    for row in df.itertuples():
        user_sequences[row.user_id].append(row.item_id)
    # Filter users with short histories
    return {u: s for u, s in user_sequences.items() if len(s) >= min_len}

def load_data(min_len=6):
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    # Build user -> sequence of item interactions
    user_sequences = create_user_sequences(df, min_len=min_len)

    # Get total item count for action space
    item_ids = set()
    for seq in user_sequences.values():
        item_ids.update(seq)
    item_count = max(item_ids)

    return user_sequences, item_count
