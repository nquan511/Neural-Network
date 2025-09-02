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
# Data
# --------------------
closes = pd.read_csv(
    R'D:\Quan\Quants\Neural Network\financial_attention\usd_1h_data_20250101_20250601.csv',
    index_col=0, parse_dates=True
)[['SOL','ETH','BTC']]

# use log-returns (safer for modeling)
rets = np.log(closes).diff().dropna()
n_assets = rets.shape[1]

data = torch.tensor(rets.values, dtype=torch.float32)

n = int(0.9*len(data)) # train/val split
train_data = data[:n]
val_data = data[n:]

# --------------------
# Batch loader
# --------------------
target_asset_idx = 0   # choose target asset (0=SOL, 1=ETH, 2=BTC)

# Count number of trainable parameters in your model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batch(split='train'):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size - 1, (batch_size,))

    # autoregressive history up to t-1
    hist = torch.stack([dataset[i:i+block_size] for i in ix])   # (B,T,n_assets)

    # contemporaneous info at time t (all assets except target)
    others_t = torch.stack([dataset[i+1:i+block_size+1].clone() for i in ix])  # (B,T,n_assets)
    others_t = torch.cat([
        others_t[:,:, :target_asset_idx],
        others_t[:,:, target_asset_idx+1:]
    ], dim=2)  # drop target

    # target value at time t
    target = torch.stack([dataset[i+1:i+block_size+1, target_asset_idx] for i in ix])  # (B,T)

    return hist.to(device), others_t.to(device), target.to(device)


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
            nn.GELU(approximate='tanh'), # Should we use Tanh activation here? squash values to be between -1 and 1
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
# Single-Asset Predictor
# --------------------
class TimeSeriesSingleAssetGPT(nn.Module):
    def __init__(self, target_idx):
        super().__init__()
        self.target_idx = target_idx
        self.input_proj = nn.Linear(n_assets, n_embd)
        self.pos_emb = SinusoidalEmbedding(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        # contemporaneous conditioning
        self.cond_proj = nn.Linear(n_assets-1, n_embd)

        # output: scalar (target asset)
        self.head = nn.Linear(n_embd, 1) # replacing self.lm_head where it maps n_embd to vocab_size in original GPT

    def forward(self, hist, others_t, target=None):
        B, T, _ = hist.shape
        x = self.input_proj(hist)
        pos = torch.arange(T, device=hist.device)
        x = x + self.pos_emb(pos)[None, :, :]
        x = self.blocks(x)
        x = self.ln_f(x)

        # last hidden state (autoregressive context)
        # h = x[:, -1, :]

        # add contemporaneous conditioning
        cond = self.cond_proj(others_t)
        # h = h + cond
        h = x + cond 

        out = self.head(h).squeeze(-1)  # (B,)

        loss = None
        if target is not None:
            loss = F.mse_loss(out, target)
        return out, loss


# --------------------
# Training
# --------------------
model = TimeSeriesSingleAssetGPT(target_asset_idx).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(1000):
    hist, others_t, target = get_batch('train')
    preds, loss = model(hist, others_t, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}: loss {loss.item():.6f}")

# --------------------
# Validation
# --------------------
model.eval()
with torch.no_grad():
    hist, others_t, target = get_batch('val')
    preds, val_loss = model(hist, others_t, target)
    print(f"Validation loss: {val_loss.item():.6f}")

# --------------------
# Plot results
# --------------------
asset_names = closes.columns.tolist()
print(f"Trainable parameters: {count_parameters(model):,}")
plt.figure(figsize=(12, 2 * preds.shape[0]))
for i in range(preds.shape[0]):  # Loop over batch dimension
    plt.subplot(preds.shape[0], 1, i + 1)
    plt.plot(target[i].cpu().numpy(), color='blue', label='Actual')
    plt.plot(preds[i].cpu().numpy(), color='orange', linestyle='--', label='Predicted')
    plt.title(f'Sample {i+1}')
    if i == 0:
        plt.legend()
    plt.ylabel('Log Return')
plt.xlabel('Time Step')
plt.suptitle(f'Predicted vs Actual {asset_names[target_asset_idx]} Returns (Validation Batch)')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
