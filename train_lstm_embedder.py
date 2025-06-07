import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import LSTMEmbedder
from movielens_loader import load_data
from tqdm import tqdm
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NextItemDataset(Dataset):
    def __init__(self, user_sequences, seq_len=5):
        self.seqs, self.targets = [], []
        for seq in user_sequences.values():
            for i in range(seq_len, len(seq)):
                self.seqs.append(seq[i-seq_len:i])
                self.targets.append(seq[i])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx): 
        return torch.tensor(self.seqs[idx]), torch.tensor(self.targets[idx])

def train_lstm_embedder(epochs=79, seq_len=20, batch_size=128, embedding_dim=64, hidden_dim=128):
    user_sequences, item_count = load_data()
    dataset = NextItemDataset(user_sequences, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMEmbedder(item_count, embedding_dim, hidden_dim).to(device)
    with torch.no_grad():
        dummy_input = torch.randint(0, item_count, (1, seq_len)).to(device)
        emb_size = model(dummy_input).shape[-1]

    predictor = nn.Linear(emb_size, item_count + 1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-3)

    checkpoint_path = "lstm_embedder.pt"
    start_epoch = 0

    # Initialize metrics dictionary
    metrics = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'best_loss': float('inf'),
        'best_accuracy': 0.0
    }

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
        print(f"🔄 Loaded checkpoint from epoch {start_epoch}")

    model.train()
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            emb = model(batch_x)
            logits = predictor(emb)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch_y).sum().item()
            total_predictions += batch_y.size(0)

        # Calculate epoch metrics
        epoch_loss = total_loss / len(loader)
        epoch_accuracy = correct_predictions / total_predictions
        
        # Store metrics
        metrics['epoch_losses'].append(epoch_loss)
        metrics['epoch_accuracies'].append(epoch_accuracy)
        
        # Update best metrics
        if epoch_loss < metrics['best_loss']:
            metrics['best_loss'] = epoch_loss
        if epoch_accuracy > metrics['best_accuracy']:
            metrics['best_accuracy'] = epoch_accuracy

        if(epoch%100==0):
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")

        # Save checkpoint with metrics
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path)

    # Save final metrics
    with open('lstm_embedder_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    torch.save(model.state_dict(), "lstm_embedder_final.pt")
    print("✅ LSTM embedder fully trained and saved to `lstm_embedder_final.pt`")
    print("📊 Training metrics saved to `lstm_embedder_metrics.json`")
    return model

if __name__ == "__main__":
    train_lstm_embedder()
