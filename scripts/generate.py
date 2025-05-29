import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from models.model import Microformer
from tokenizers import Tokenizer
from config import VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN, ADAPTER_DIM
import sqlite3
from datetime import datetime

# --- Load tokenizer and model ---
tokenizer = Tokenizer.from_file("data/tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

model = Microformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    long_term_adapter_dim=ADAPTER_DIM,
    session_adapter_dim=ADAPTER_DIM
)
model.load_state_dict(torch.load("microformer.pt"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Freeze all but session adapters and output for online learning ---
model.freeze_except_adapters(session_only=True, include_output=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-2  # High LR for visible learning during teaching
)

# --- Memory DB setup ---
conn = sqlite3.connect("memory.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS memory (
    timestamp TEXT,
    prompt TEXT,
    response TEXT
)
""")
conn.commit()

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    logits = logits.squeeze(0)  # [1, vocab] → [vocab]
    probs = torch.softmax(logits, dim=-1)

    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Top-p mask
    sorted_mask = cumulative_probs > top_p
    sorted_mask[1:] = sorted_mask[:-1].clone()
    sorted_mask[0] = False

    # Top-k mask
    if top_k < sorted_probs.size(0):
        sorted_mask[top_k:] = True

    # Zero out masked values
    sorted_probs[sorted_mask] = 0.0

    # Normalize and sample
    sorted_probs /= sorted_probs.sum()
    sampled_relative_index = torch.multinomial(sorted_probs, 1).item()
    sampled_token_id = sorted_indices[sampled_relative_index].item()

    return sampled_token_id

def generate(prompt, length=100, temperature=1.0, top_p=0.9, top_k=50):
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.token_to_id("<EOS>")

    for _ in range(length):
        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            for token_id in input_tensor[0].tolist():
                logits[0, token_id] *= 0.8

            next_token_id = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)

        if next_token_id == eos_token_id:
            break

    output_ids = input_tensor[0].tolist()
    decoded = tokenizer.decode(output_ids)

    if "<EOS>" in decoded:
        decoded = decoded.split("<EOS>")[0].strip()

    return decoded

def online_unsupervised_update(model, tokenizer, text, optimizer, loss_fn, device, max_len=64):
    # Always called after freeze_except_adapters(session_only=True)
    ids = tokenizer.encode(text).ids + [tokenizer.token_to_id("<EOS>")]
    if len(ids) < 2:
        return None

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

# Optional: Reset session adapter weights between sessions
def reset_session_adapters(model):
    for layer in model.layers:
        if getattr(layer, 'session_adapter', None) is not None:
            for param in layer.session_adapter.parameters():
                if param.data is not None:
                    nn.init.zeros_(param.data)

if __name__ == "__main__":
    while True:
        prompt = input("\nEnter a prompt (or 'exit' to quit): ")
        if prompt.lower() in {"exit", "quit"}:
            break
        temp = float(input("Temperature (e.g. 0.7, 1.0): "))

        output = generate(prompt, length=100, temperature=temp, top_p=0.9, top_k=50)
        print("\nGenerated text:\n")
        print(output)

        # Online learning: always update session adapters only
        teach = input("\nDo you want to teach the model a better answer? (y/N): ").strip().lower()
        if teach == "y":
            your_answer = input("Type your ideal response for this prompt: ")
            model.freeze_except_adapters(session_only=True, include_output=True)
            online_text = prompt + " " + your_answer
            loss = online_unsupervised_update(
                model, tokenizer, online_text, optimizer, criterion, device, max_len=MAX_SEQ_LEN
            )
            print(f"[Online update loss: {loss:.4f}]")
        else:
            model.freeze_except_adapters(session_only=True, include_output=True)
            online_text = prompt + " " + output
            loss = online_unsupervised_update(
                model, tokenizer, online_text, optimizer, criterion, device, max_len=MAX_SEQ_LEN
            )
            print(f"[Online (self-improve) update loss: {loss:.4f}]")

        # Store the interaction in memory DB as before
        c.execute("INSERT INTO memory (timestamp, prompt, response) VALUES (?, ?, ?)",
                  (datetime.now().isoformat(timespec='seconds'), prompt, output))
        conn.commit()

        print("\nRecent memory:")
        for row in c.execute("SELECT * FROM memory ORDER BY timestamp DESC LIMIT 5"):
            print(f"[{row[0]}] {row[1]} → {row[2]}")

        # Optional: Uncomment to reset fast-memory (session adapters) between users/sessions
        # reset_session_adapters(model)
