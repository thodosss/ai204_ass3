import torch.nn as nn
import torch
class LSTMEmbedder(nn.Module):
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(x)  # (1, batch_size, hidden_dim)
        return h_n.squeeze(0)       # (batch_size, hidden_dim)


