# Custom SML

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Custom SML** is a custom PyTorch implementation of a Small Language Model (SML) inspired by the Gemma architecture. It is designed to be trained from scratch and is configured here to learn from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

## ✨ Features

- **Gemma-Inspired Architecture**: Implements advanced transformer features including Grouped Query Attention (GQA) and Rotary Positional Embeddings (RoPE).
- **Sliding Window Attention**: Combines full attention and sliding window attention layers for efficient long-context modeling.
- **Efficient Normalization**: Uses `RMSNorm` for stable and efficient training.
- **Optimized Training Loop**: Includes Mixed Precision Training (`bfloat16`), Gradient Accumulation, and a Cosine Annealing Learning Rate Scheduler with Warmup.
- **Custom Tokenization**: Uses standard Byte-Pair Encoding (BPE) via `tiktoken` (GPT-2 encoding).

## 📊 Architecture Configuration (`270M` profile)

- **Parameters**: ~270M
- **Vocabulary Size**: 50,257
- **Layers**: 18 (Mix of `sliding_attention` and `full_attention`)
- **Embedding Dimension**: 640
- **Heads**: 4 (Query Heads), 1 (KV Group for GQA)
- **Context Length**: Up to 32,768 tokens (Sliding Window: 512)

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch numpy datasets tiktoken tqdm
```

### 2. Data Preparation

The model uses the `TinyStories` dataset. The training script automatically downloads and tokenizes the dataset using `tiktoken`. It saves the tokenized data into `.bin` memory-mapped files (`train.bin`, `validation.bin`) for highly efficient batched loading.

### 3. Training

The training loop is completely custom and tracks validation loss, periodically saving the best model state to `best_model_params.pt`.

Key training hyperparameters (configurable in the code):
- **Batch Size**: 32
- **Block Size**: 128
- **Learning Rate**: 1e-4 (Peak)
- **Gradient Accumulation Steps**: 32

### 4. Inference / Generation

To generate text using a trained model, utilize the `generate` function:

```python
import torch

# Load tokenzier
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Load model
model = Gemma3Model(GEMMA3_CONFIG_270M)
model.load_state_dict(torch.load("best_model_params.pt"))
model.to("cuda")
model.eval()

# Generate
sentence = "There is a tree in the forest "
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0).to("cuda")
output_ids = model.generate(context, max_new_tokens=200)

print(enc.decode(output_ids.squeeze().tolist()))
```

## 📁 Repository Structure

- `Gemma.ipynb`: Main Jupyter Notebook containing the full implementation, dataset processing, model architecture, training loop, and inference code.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
