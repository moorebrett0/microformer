import torch
import json
import numpy
from tokenizers import Tokenizer
from pathlib import Path

# Load tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

# Load corpus
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encode with BPE tokenizer
encoded = tokenizer.encode(text).ids

# Convert to tensor and split into train/val
data = torch.tensor(encoded, dtype=torch.long)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# Save outputs
torch.save(train_data, "data/train.pt")
torch.save(val_data, "data/val.pt")
