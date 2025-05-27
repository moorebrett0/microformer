import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import json
import torch.nn.functional as F
from models.model import Microformer
from config import *
from scripts.memory import save_memory, recall_memories
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

# Load model
model = Microformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN)
model.load_state_dict(torch.load("microformer.pt"))
model.eval()

# Encode and decode using tokenizer
def encode(s):
    return tokenizer.encode(s).ids

def decode(l):
    return tokenizer.decode(l)

# Top-k filtering
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., -1, None]] = -float("Inf")
    return out

# Generate function with top-k and repeat filtering
def generate(prompt, length=100, temperature=1.0, top_k=10):
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)
    generated = input_ids.tolist()[0]

    for _ in range(length):
        x = input_ids[:, -MAX_SEQ_LEN:]
        with torch.no_grad():
            logits = model(x)
        next_token_logits = logits[0, -1, :]

        # Apply top-k filtering and temperature
        filtered_logits = top_k_logits(next_token_logits, k=top_k)
        probs = F.softmax(filtered_logits / temperature, dim=0)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Avoid repeating the same token 3 times in a row
        if len(generated) > 2 and next_token == generated[-1] == generated[-2]:
            continue

        generated.append(next_token)
        input_ids = torch.tensor([generated], dtype=torch.long)

    return decode(generated)

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    temp = input("Temperature (e.g. 0.7, 1.0): ")
    try:
        temperature = float(temp)
    except:
        temperature = 1.0

    output = generate(prompt, length=200, temperature=temperature)

    print("\nGenerated text:\n")
    print(output)

    save_memory(prompt, output)

    print("\nRecent memory:")
    for p, r, t in recall_memories():
        print(f"[{t}] {p} → {r}")
