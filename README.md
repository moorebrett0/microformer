# Microformer

**Microformer** is a minimal, educational-scale transformer language model built from scratch in PyTorch.  
Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and OpenAI’s GPT-1, Microformer is designed for learning, experimentation, and prototyping on lightweight datasets like [text8](https://mattmahoney.net/dc/textdata.html) or Tiny Shakespeare.

---

## Features

- Decoder-only transformer (GPT-style) architecture
- Choice of character-level **or** subword/BPE tokenization (configurable)
- Learnable positional encoding
- Multi-head self-attention
- Configurable depth, embedding size, sequence length, and attention heads
- Simple end-to-end pipeline: preprocessing, training, and text generation
- Modular, readable code ideal for educational use and tinkering
- Temperature and multinomial sampling in text generation

---

## Project Structure

```
microformer/
├── config.py              # Hyperparameters and model settings
├── data/
│   ├── corpus.txt         # Raw training text
│   ├── train.pt           # Preprocessed training tensor (token IDs)
│   ├── val.pt             # Validation tensor (token IDs)
│   ├── vocab.json         # Vocabulary (char or subword, stoi/itos mapping)
│   └── tokenizer.json     # (optional) BPE tokenizer file if using subwords
├── models/
│   └── model.py           # Transformer model definition (Microformer)
├── scripts/
│   ├── prepare_data.py    # Data preprocessing/tokenization
│   ├── train.py           # Training script
│   └── generate_text.py   # Inference/generation script
│   └── tokenizer_setup.py # BPE Tokenizer
└── README.md
```

---

## Quickstart

1. **Load your corpus and run tokenizer**

   First, make sure you have a training corpus ready at `data/corpus.txt`. This should be a plain text file containing the data you want to train your microformer on (for example: sentences, phrases, or any text lines—one per line).

2. **Choose your tokenizer:**

- **Character-level (default):**  
  No extra steps needed—just run the main data prep script.

- **BPE/Subword (optional, recommended for larger or more complex text):**  
  Set up BPE vocab with:
  ```bash
  python scripts/tokenizer_setup.py --input data/corpus.txt --vocab_size 1000
  
  Adjust --vocab_size as desired.


3. **Prepare the dataset**

   ```bash
   python scripts/prepare_data.py
   ```
   - Reads `data/corpus.txt`
   - Trains a vocabulary/tokenizer (char-level or BPE)
   - Encodes text as token IDs and saves `train.pt` / `val.pt`
   - Saves vocabulary as `vocab.json` (and `tokenizer.json` for BPE)

4. **Train the model**

   ```bash
   python scripts/train.py
   ```
   - Loads tokenized data and vocabulary
   - Configures model via `config.py`
   - Trains a transformer on next-token prediction

5. **Generate text**

   ```bash
   python scripts/generate_text.py
   ```
   - Loads a trained checkpoint
   - Prompts for a seed string and temperature
   - Generates new text in the style of your corpus

---

## Example Config (`config.py`)

```python
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
VOCAB_SIZE = 100  # Set automatically from tokenizer/vocab
```

---

## Customization & Ideas

- Use BPE/subword tokenization for more expressive modeling (recommended for non-trivial datasets)
- Swap in larger datasets: `text8`, `tinyshakespeare`, etc.
- Add validation loss, checkpointing, or early stopping
- Visualize training with TensorBoard or wandb
- Experiment with alternative attention mechanisms or memory modules

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [tokenizers](https://github.com/huggingface/tokenizers) (for BPE/subword)

Install dependencies with:
```bash
pip install torch tokenizers
```

---

## Credits

- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy
- Built using concepts from the original [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

---

## License

MIT License – Use freely for learning and experimentation.

---

**Happy tinkering with transformers!**
