from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path
import json

# Paths
corpus_path = Path("data/corpus.txt")
tokenizer_path = Path("data/tokenizer.json")

# Read corpus
with corpus_path.open("r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Initialize tokenizer with BPE model
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                              pre_tokenizers.Whitespace(),
                              pre_tokenizers.Punctuation()
                          ])

# Train tokenizer
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["<PAD>", "<UNK>", "<EOS>"])
tokenizer.train_from_iterator(lines, trainer)

# Save tokenizer
tokenizer.save(str(tokenizer_path))

# Create vocab.json for compatibility
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {v: k for k, v in vocab.items()}

with open("data/vocab.json", "w") as f:
    json.dump({"stoi": stoi, "itos": itos}, f)
