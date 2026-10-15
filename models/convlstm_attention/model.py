import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    """Computes temporal attention weights over a sequence of spatial features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq, hidden_dim)
        weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1), weights

class ConvLSTMAttention(nn.Module):
    """
    Phase 7: ConvLSTM + Attention Hybrid.
    Preserves spatial info across temporal sequences, applying spatial-temporal attention.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMAttention, self).__init__()
        # Simplified ConvLSTM placeholder architecture for academic prototype
        self.spatial_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=1)
        self.lstm = nn.LSTM(hidden_channels * 64 * 64, 256, batch_first=True) # assuming 64x64 patches flattened
        self.attention = TemporalAttention(256)
        self.classifier = nn.Linear(256, 4)
        
    def forward(self, x_seq):
        # x_seq shape: (batch, seq, C, H, W)
        batch, seq, C, H, W = x_seq.shape
        x_flat = x_seq.view(batch * seq, C, H, W)
        spatial_features = self.spatial_conv(x_flat)
        
        # Flatten spatial dims for temporal processing
        temporal_input = spatial_features.view(batch, seq, -1)
        lstm_out, _ = self.lstm(temporal_input)
        
        # Apply attention
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attn_weights
