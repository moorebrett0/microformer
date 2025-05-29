import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import json
from models.model import Microformer
from config import *

# ------------------------
# LOAD DATA AND VOCAB
# ------------------------
with open("data/vocab.json", "r") as f:
    vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
VOCAB_SIZE = len(stoi)

data = torch.load("data/train.pt")
SEQ_LEN = MAX_SEQ_LEN
BATCH_SIZE = 32

# Drop remainder for clean batch shape
num_batches = len(data) // (SEQ_LEN * BATCH_SIZE)
trimmed_len = num_batches * SEQ_LEN * BATCH_SIZE
data = data[:trimmed_len]
data = data.view(BATCH_SIZE, -1)  # shape: (BATCH_SIZE, n_chunks)

def get_batch(start_idx):
    x = data[:, start_idx:start_idx+SEQ_LEN]
    y = data[:, start_idx+1:start_idx+SEQ_LEN+1]
    return x, y

# ------------------------
# DEVICE SETUP
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# MODEL INSTANTIATION (with stacked adapters)
# ------------------------
model = Microformer(
    VOCAB_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    FF_DIM,
    NUM_LAYERS,
    MAX_SEQ_LEN,
    long_term_adapter_dim=ADAPTER_DIM,     # <-- set in config
    session_adapter_dim=ADAPTER_DIM        # <-- set in config
)
model.to(device)

# ------------------------
# TRAIN LONG-TERM ADAPTERS ONLY
# ------------------------
model.freeze_except_adapters(session_only=False, include_output=True)
# (Optionally, explicitly freeze session adapters:)
for layer in model.layers:
    if getattr(layer, 'session_adapter', None) is not None:
        for param in layer.session_adapter.parameters():
            param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# ------------------------
# MAIN BATCH TRAINING LOOP (CORPUS)
# ------------------------
for epoch in range(6):
    for i in range(0, data.shape[1] - SEQ_LEN, SEQ_LEN):
        inputs, targets = get_batch(i)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "microformer.pt")

# ------------------------
# ONLINE (SESSION) LEARNING UTILITY
# ------------------------
def online_unsupervised_update(model, tokenizer, text, optimizer, loss_fn, device, max_len=64):
    # Only update session adapters/output layer; call freeze_except_adapters before this as needed.
    ids = tokenizer.encode(text).ids + [tokenizer.token_to_id("<EOS>")]
    if len(ids) < 2:
        return None  # not enough tokens

    ids = ids[:max_len + 1]
    input_ids = ids[:-1]
    target_ids = ids[1:]
    input_ids += [tokenizer.token_to_id("<PAD>")] * (max_len - len(input_ids))
    target_ids += [tokenizer.token_to_id("<PAD>")] * (max_len - len(target_ids))
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)

    model.train()
    logits = model(input_tensor)
    logits = logits.view(-1, logits.size(-1))
    targets = target_tensor.view(-1)
    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()

# ------------------------
# SESSION ADAPTER RESET FUNCTION (OPTIONAL)
# ------------------------
def reset_session_adapters(model):
    for layer in model.layers:
        if getattr(layer, 'session_adapter', None) is not None:
            for param in layer.session_adapter.parameters():
                if param.data is not None:
                    nn.init.zeros_(param.data)

# ------------------------
# USAGE FOR ONLINE LEARNING (after chat, NOT in main batch loop):
# ------------------------
# from tokenizers import Tokenizer
# tokenizer = Tokenizer.from_file("data/tokenizer.json")
# model.freeze_except_adapters(session_only=True, include_output=True)
# message = "Who is Buck?"
# loss = online_unsupervised_update(model, tokenizer, message, optimizer, criterion, device, max_len=SEQ_LEN)
# print(f"Online update loss: {loss}")

