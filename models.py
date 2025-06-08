import torch
import torch.nn as nn

class LSTMEmbedder(nn.Module):
    def __init__(self, item_count, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.2, genre_dim=0):
        super(LSTMEmbedder, self).__init__()
        self.embedding = nn.Embedding(item_count + 2, embedding_dim, padding_idx=0)
        self.genre_dim = genre_dim
        self.lstm = nn.LSTM(
            embedding_dim + genre_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # PyTorch only applies dropout if num_layers > 1
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x, genre_feats=None):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        if genre_feats is not None and self.genre_dim > 0:
            x = torch.cat([x, genre_feats], dim=-1)  # (batch, seq_len, embedding_dim + genre_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output
        out = self.dropout(out)
        return out


