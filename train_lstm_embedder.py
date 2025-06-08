import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import LSTMEmbedder
from movielens_loader import load_data, split_user_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NextItemDataset(Dataset):
    def __init__(self, user_sequences, seq_len=5, item_genre_dict=None, genre_dim=0):
        self.samples = []
        self.item_genre_dict = item_genre_dict
        self.genre_dim = genre_dim
        for user, seq in user_sequences.items():
            for i in range(seq_len, len(seq)):
                history = seq[i-seq_len:i]
                target = seq[i]
                self.samples.append((history, target))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx): 
        history, target = self.samples[idx]
        history = torch.tensor(history, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        # Get genre features for each item in history
        if self.item_genre_dict is not None and self.genre_dim > 0:
            genre_feats = []
            for item in history:
                genre = self.item_genre_dict.get(int(item.item()), np.zeros(self.genre_dim, dtype=np.float32))
                genre_feats.append(torch.tensor(genre, dtype=torch.float32))
            genre_feats = torch.stack(genre_feats)  # shape: (seq_len, genre_dim)
        else:
            genre_feats = torch.zeros((self.seq_len, self.genre_dim), dtype=torch.float32)
        return history, genre_feats, target

def split_user_sequences(user_sequences, val_ratio=0.1):
    train_seqs, val_seqs = {}, {}
    for user, seq in user_sequences.items():
        n = len(seq)
        n_val = int(n * val_ratio)
        train_seqs[user] = seq[:-n_val] if n_val > 0 else seq
        val_seqs[user] = seq[-n_val:] if n_val > 0 else []
    return train_seqs, val_seqs

def train_lstm_embedder(epochs=79, seq_len=20, batch_size=128, embedding_dim=64, hidden_dim=128, patience=30, dropout=0.4):
    # Load genre info
    item_path = "ml-100k/u.item"
    genre_cols = [
        "movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    item_df = pd.read_csv(item_path, sep='|', names=genre_cols, encoding='latin-1')
    genre_columns = genre_cols[5:]
    item_genre_dict = {
        int(row['movie_id']): row[genre_columns].values.astype(np.float32)
        for _, row in item_df.iterrows()
    }
    genre_dim = len(genre_columns)

    user_sequences, item_count = load_data()
    train_sequences, val_sequences = split_user_sequences(user_sequences, val_ratio=0.1)

    train_dataset = NextItemDataset(train_sequences, seq_len, item_genre_dict, genre_dim)
    val_dataset = NextItemDataset(val_sequences, seq_len, item_genre_dict, genre_dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMEmbedder(item_count, embedding_dim, hidden_dim, dropout=dropout, genre_dim=genre_dim).to(device)
    predictor = nn.Linear(hidden_dim, item_count + 1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-4)

    checkpoint_path = "lstm_embedder.pt"
    start_epoch = 0

    # Initialize metrics dictionary
    metrics = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'best_loss': float('inf'),
        'best_accuracy': 0.0
    }

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'predictor_state_dict' in checkpoint:
            predictor.load_state_dict(checkpoint['predictor_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        metrics = checkpoint['metrics']

    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        model.train()
        predictor.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for histories, genre_feats, targets in train_loader:
            histories, genre_feats, targets = histories.to(device), genre_feats.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(histories, genre_feats)
            logits = predictor(outputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
        avg_train_loss = train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation
        model.eval()
        predictor.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for histories, genre_feats, targets in val_loader:
                histories, genre_feats, targets = histories.to(device), genre_feats.to(device), targets.to(device)
                outputs = model(histories, genre_feats)
                logits = predictor(outputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        avg_val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        metrics['epoch_losses'].append(avg_train_loss)
        metrics['epoch_accuracies'].append(train_acc)
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_accuracies'].append(val_acc)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in val_loss.")
                break

    # Save final metrics
    with open('lstm_embedder_metrics.json', 'w') as f:
        import json
        json.dump(metrics, f)

    torch.save(model.state_dict(), "lstm_embedder_final.pt")
    return model


if __name__ == "__main__":
    train_lstm_embedder()