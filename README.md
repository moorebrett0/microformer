# Microformer

**Microformer** is a minimal, educational-scale transformer language model built from scratch in PyTorch.  
Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and OpenAI’s GPT-1, Microformer is designed for learning, experimentation, and prototyping on lightweight datasets like [text8](https://mattmahoney.net/dc/textdata.html) or Tiny Shakespeare.

---

## Features

- Decoder-only transformer (GPT-style) architecture
- **Stacked adapters per layer for dual-memory:**
    - **Long-term adapters** (for corpus/knowledge facts)
    - **Session adapters** (for rapid, online, user/session-specific learning)
- Choice of character-level **or** subword/BPE tokenization (configurable)
- Learnable positional encoding
- Multi-head self-attention
- Configurable depth, embedding size, sequence length, and attention heads
- Simple end-to-end pipeline: preprocessing, training, and text generation
- Modular, readable code ideal for educational use and tinkering
- Temperature and multinomial sampling in text generation

---

## What’s Unique: Stacked Adapters for Dual-Memory Learning

Microformer implements **two adapters in every transformer block**:

- **Long-term adapter:**  
  Trained with your full corpus during batch/corpus training.  
  Stores stable, general “knowledge” (e.g., literary style, factual info).

- **Session adapter:**  
  Starts blank and is trained *on the fly* during chat or interactive teaching.  
  Lets you rapidly “teach” new facts, styles, or user preferences without overwriting core knowledge.

At inference, the outputs of both adapters (plus the core transformer) are combined—giving the model both stable and flexible, session-specific memory, just like a human brain’s “temporal lobe” and “core memory”.

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
│   ├── train.py           # Training script (trains long-term adapters)
│   ├── generate_text.py   # Inference/generation + online learning (session adapters)
│   └── tokenizer_setup.py # BPE Tokenizer
└── README.md
```

---

## Quickstart

1. **Prepare your corpus and run the tokenizer**

   Place your text data in `data/corpus.txt`.

2. **Choose your tokenizer:**

- **Character-level (default):**  
  No extra steps needed.

- **BPE/Subword (recommended for rich/modern text):**
  ```bash
  python scripts/tokenizer_setup.py --input data/corpus.txt --vocab_size 1000
  ```

3. **Prepare the dataset**

   ```bash
   python scripts/prepare_data.py
   ```

4. **Train the model (long-term knowledge)**

   ```bash
   python scripts/train.py
   ```
    - This trains only the **long-term adapters** and core weights.
    - Session adapters remain untrained (blank) until chat time.

5. **Generate text and teach interactively (session memory)**

   ```bash
   python scripts/generate_text.py
   ```
    - Loads your trained model.
    - Prompts for a seed string and temperature.
    - **Allows you to “teach” new facts on the fly!**
    - New knowledge is stored in session adapters—does *not* overwrite long-term knowledge.

---

## Example Config (`config.py`)

```python
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
ADAPTER_DIM = 32   # Used for both long-term and session adapters
VOCAB_SIZE = 100   # Set automatically from tokenizer/vocab
```

---

## Using the Dual-Memory System

- **Long-term adapters:**  
  Learned during `train.py`—persist between runs.

- **Session adapters:**  
  Learned during interactive chat in `generate_text.py`—resettable (optional) between users/sessions.

- **Teach new facts by entering a prompt and providing your ideal answer.**  
  The model will “remember” this during the session, even if it wasn’t present in the training corpus.

---

## Customization & Ideas

- Use BPE/subword tokenization for more expressive modeling (recommended for non-trivial datasets)
- Add more adapters or experiment with gating (e.g., blend adapters by context)
- Combine with a key-value retrieval or buffer for truly persistent “user memory”
- Visualize training with TensorBoard or wandb
- Tinker with alternative attention or memory mechanisms

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
- Adapter and continual-learning inspiration from recent NLP research ([Houlsby et al. 2019](https://arxiv.org/abs/1902.00751))
- Built using concepts from the original [GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

---

## License

MIT License – Use freely for learning and experimentation.

---

**Happy tinkering with dual-memory transformers!**
