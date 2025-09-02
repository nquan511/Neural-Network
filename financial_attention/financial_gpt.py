import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# --------------------
# Hyperparameters
# --------------------
batch_size = 32
block_size = 30   # context length (days)
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# --------------------
# Dummy data (replace with real returns)
# --------------------
closes = pd.read_csv(R'D:\Quan\Quants\Neural Network\financial_attention\usd_1h_data_20250101_20250601.csv', index_col=0, parse_dates=True)[['SOL','ETH','BTC']]
n_assets = len(closes.columns)
# Compute log-returns
data = torch.tensor(
    closes.values,
    dtype=torch.float32
)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])  # (B, T, n_assets)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # shifted by 1
    x = x if x.ndim > 1 else x.unsqueeze(-1)
    y = y if y.ndim > 1 else y.unsqueeze(-1)
    return x.to(device), y.to(device)

# --------------------
# Model components
# --------------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]



class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --------------------
# GPT for Financial Returns
# --------------------
class TimeSeriesGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Project input returns into embedding space
        self.input_proj = nn.Linear(n_assets, n_embd)

        # Positional embeddings
        self.pos_emb = SinusoidalEmbedding(n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        # Final regression head: predict next-step returns
        self.head = nn.Linear(n_embd, n_assets)

    def forward(self, idx, targets=None):
        B, T,_ = idx.shape
        x = self.input_proj(idx)  # (B,T,n_embd)
        pos = torch.arange(T, device=idx.device)
        x = x + self.pos_emb(pos)[None, :, :]  # add positional encoding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,n_assets)

        if targets is None:
            loss = None
        else:
            loss = F.mse_loss(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """autoregressive rollout for forecasting"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            next_val = logits[:, -1, :]  # predicted next-day returns
            idx = torch.cat((idx, next_val.unsqueeze(1)), dim=1)
        return idx

# --------------------
# Training loop
# --------------------
model = TimeSeriesGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(1000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step}: loss {loss.item():.6f}")

# --------------------
# Validation - Predicting and Plotting Returns
# --------------------
# Get a batch from the validation set
model.eval()
with torch.no_grad():
    xb, yb = get_batch('val')
    preds, val_loss = model(xb,yb)
    print(f"Validation loss: {val_loss.item():.6f}")

# Convert to numpy for plotting
# We'll plot the first sample in the batch and the first asset
asset_names = closes.columns.tolist()

plt.figure(figsize=(10, 5))
for i, asset in enumerate(asset_names):
    plt.plot(yb[0, :, i].cpu().numpy(), label=f'Actual {asset}')
    plt.plot(preds[0, :, i].cpu().numpy(), '--', label=f'Predicted {asset}')
plt.title('Predicted vs Actual Returns (Validation Set)')
plt.xlabel('Time Step')
plt.ylabel('Log Return')
plt.legend()
plt.show()
