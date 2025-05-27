# Microformer

**Microformer** is a minimal, educational-scale transformer language model built from scratch in PyTorch. Inspired by projects like [minGPT](https://github.com/karpathy/minGPT) and OpenAI's original GPT-1, this project is ideal for understanding transformer internals, experimenting with new ideas, and training on lightweight datasets like [text8](https://mattmahoney.net/dc/textdata.html) or Tiny Shakespeare.

---

## Features

- Decoder-only transformer architecture  
- Character-level or subword tokenization (char-level default)  
- Positional encoding  
- Multi-head self-attention  
- Configurable depth, width, and sequence length  
- Simple training loop with PyTorch  
- Inference/generation script with temperature sampling  

---

## Project Structure

```
microformer/
├── config.py            # Model hyperparameters
├── data/
│   ├── corpus.txt       # Your training text
│   ├── train.pt         # Preprocessed training tensor
│   ├── val.pt           # Validation tensor (optional)
│   └── vocab.json       # Character vocabulary mapping
├── models/
│   └── model.py         # Transformer model definition
├── scripts/
│   ├── prepare_data.py  # Data preprocessing script
│   ├── train.py         # Training loop
│   └── generate_text.py # Inference/generation script
└── README.md
```

---

##  Training

1. **Prepare the dataset**:

   ```bash
   python scripts/prepare_data.py
   ```

This reads `data/corpus.txt`, builds a vocabulary, encodes it to token IDs, and saves `train.pt`/`val.pt`.

---

2. **Train the model**:

   ```bash
   python scripts/train.py
   ```

   You can configure model size, batch size, sequence length, and epochs via `config.py`.

3. **Generate text**:

   ```bash
   python scripts/generate_text.py
   ```

   This loads a saved model and prompts for a seed string + temperature.

---

##  Example Config (`config.py`)

```python
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
MAX_SEQ_LEN = 128
VOCAB_SIZE = 100  # Overridden at runtime
```

---

##  Ideas to Extend

- Switch to BPE or WordPiece tokenization
- Add validation loss and early stopping
- Train on `text8`, `tinyshakespeare`, or `OpenWebText`
- Replace softmax with sparse attention
- Explore memory-augmented architectures
- Log loss/attention via TensorBoard or wandb

---

## ️ Requirements

- Python 3.8+
- PyTorch

Install with:

```bash
pip install torch
```

---

##  Credits

- Inspired by [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy
- Based on concepts from the original [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

---

##  License

MIT License. Use freely for learning and experimentation.


