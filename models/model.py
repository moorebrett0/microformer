import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.ff(x)

# --- NEW: Adapter Block ---
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.relu = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)
    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))  # Residual

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim,
                 long_term_adapter_dim=None, session_adapter_dim=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Two adapters: one for long-term (rarely updated), one for session (online)
        self.long_term_adapter = Adapter(embed_dim, long_term_adapter_dim) if long_term_adapter_dim else None
        self.session_adapter = Adapter(embed_dim, session_adapter_dim) if session_adapter_dim else None

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        # Add both adapters’ outputs, if present
        if self.long_term_adapter is not None:
            x = self.long_term_adapter(x)
        if self.session_adapter is not None:
            x = self.session_adapter(x)
        return x


class Microformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len,
                 long_term_adapter_dim=None, session_adapter_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, ff_dim,
                long_term_adapter_dim=long_term_adapter_dim,
                session_adapter_dim=session_adapter_dim
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

    def freeze_except_adapters(self, session_only=True, include_output=True):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.layers:
            if getattr(layer, 'session_adapter', None) is not None:
                for param in layer.session_adapter.parameters():
                    param.requires_grad = True
            if not session_only and getattr(layer, 'long_term_adapter', None) is not None:
                for param in layer.long_term_adapter.parameters():
                    param.requires_grad = True
        if include_output:
            for param in self.output.parameters():
                param.requires_grad = True

