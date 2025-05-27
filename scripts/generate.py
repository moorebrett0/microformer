import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import json
import torch.nn.functional as F
from models.model import Microformer
from config import *

# Load vocab
with open("data/vocab.json", "r") as f:
    vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
VOCAB_SIZE = len(stoi)

# Load model
model = Microformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN)
model.load_state_dict(torch.load("microformer.pt"))
model.eval()

# Encode and decode functions
def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Generation loop
def generate(prompt, length=100, temperature=1.0):
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0)
    generated = input_ids.tolist()[0]

    for _ in range(length):
        x = input_ids[:, -MAX_SEQ_LEN:]
        with torch.no_grad():
            logits = model(x)
        next_token_logits = logits[0, -1, :]

        # Sample with temperature
        probs = F.softmax(next_token_logits / temperature, dim=0)
        next_token = torch.multinomial(probs, num_samples=1).item()

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
