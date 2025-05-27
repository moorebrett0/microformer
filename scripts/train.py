import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import json
from models.model import Microformer
from config import *

# Load prepared dataset and vocab
with open("data/vocab.json", "r") as f:
    vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
VOCAB_SIZE = len(stoi)

data = torch.load("data/train.pt")
inputs = data[:-1].unsqueeze(0)   # (1, seq_len - 1)
targets = data[1:].unsqueeze(0)   # (1, seq_len - 1)

SEQ_LEN = MAX_SEQ_LEN
BATCH_SIZE = 32

# Drop remainder to keep shape clean
num_batches = len(data) // (SEQ_LEN * BATCH_SIZE)
trimmed_len = num_batches * SEQ_LEN * BATCH_SIZE
data = data[:trimmed_len]

# Reshape
data = data.view(BATCH_SIZE, -1)  # shape: (BATCH_SIZE, n_chunks)

def get_batch(start_idx):
    x = data[:, start_idx:start_idx+SEQ_LEN]
    y = data[:, start_idx+1:start_idx+SEQ_LEN+1]
    return x, y


# Instantiate model
model = Microformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(50):
    for i in range(0, data.shape[1] - SEQ_LEN, SEQ_LEN):
        inputs, targets = get_batch(i)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "microformer.pt")
