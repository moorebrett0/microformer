import torch
import json
from pathlib import Path

# Load the dummy corpus
DATA_PATH = Path("data/corpus.txt")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Create character-level vocabulary
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Save vocab
VOCAB_PATH = Path("data/vocab.json")
with open(VOCAB_PATH, "w") as f:
    json.dump({"stoi": stoi, "itos": itos}, f)

# Encode text into tensor of token IDs
def encode(s):
    return [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)

# Optionally split into train/val (simple 90/10 split)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# Save to disk
torch.save(train_data, "data/train.pt")
torch.save(val_data, "data/val.pt")

print(f"Data preparation complete! {len(train_data)} train tokens, {len(val_data)} val tokens.")
