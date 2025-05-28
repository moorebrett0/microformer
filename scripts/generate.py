import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# scripts/generate.py
import torch
import torch.nn.functional as F
from models.model import Microformer
from tokenizers import Tokenizer
from config import VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN
import sqlite3
from datetime import datetime

# Load tokenizer and model
tokenizer = Tokenizer.from_file("data/tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

model = Microformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
)
model.load_state_dict(torch.load("microformer.pt"))
model.eval()

# Setup memory DB
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

def top_p_sample(logits, p=0.9):
    probs = torch.softmax(logits, dim=-1).squeeze()
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = cumulative_probs > p
    cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False
    sorted_probs[cutoff] = 0
    sorted_probs /= sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, 1).item()
    return sorted_indices[sampled_idx]

def generate(prompt, length=100, temperature=1.0, top_p=0.9):
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    eos_token_id = tokenizer.token_to_id("<EOS>")

    for _ in range(length):
        with torch.no_grad():
            logits = model(input_tensor)
            logits = logits[:, -1, :] / temperature

            next_token_id = top_p_sample(logits, p=top_p)

        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]])], dim=1)

        if next_token_id == eos_token_id:
            break

    output_ids = input_tensor[0].tolist()
    decoded = tokenizer.decode(output_ids)

    if "<EOS>" in decoded:
        decoded = decoded.split("<EOS>")[0].strip()

    return decoded

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    temp = float(input("Temperature (e.g. 0.7, 1.0): "))

    response = generate(prompt, length=100, temperature=temp, top_p=0.9)
    print("\nGenerated text:\n")
    print(response)

    c.execute("INSERT INTO memory (timestamp, prompt, response) VALUES (?, ?, ?)",
              (datetime.now().isoformat(timespec='seconds'), prompt, response))
    conn.commit()

    print("\nRecent memory:")
    for row in c.execute("SELECT * FROM memory ORDER BY timestamp DESC LIMIT 5"):
        print(f"[{row[0]}] {row[1]} → {row[2]}")
