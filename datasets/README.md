# Datasets for Naturalistic Induction Heads Research

This directory contains datasets for studying induction heads on naturalistic data.
Data files are excluded from git due to size. Follow download instructions below.

## Dataset 1: OpenWebText (Primary)

### Overview
- **Source**: HuggingFace `Skylion007/openwebtext`
- **Size**: ~8M documents, ~40GB uncompressed
- **Format**: HuggingFace Dataset (streaming supported)
- **Task**: Natural language text for studying induction heads on web text
- **Why**: Close approximation to GPT-2's training distribution; ideal for studying
  whether induction heads behave differently on naturalistic vs random data

### Download Instructions

**Streaming (recommended - no download needed):**
```python
from datasets import load_dataset
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
for item in ds:
    text = item["text"]
    # process text...
```

**Full download:**
```python
from datasets import load_dataset
ds = load_dataset("Skylion007/openwebtext", split="train")
ds.save_to_disk("datasets/openwebtext")
```

## Dataset 2: WikiText-103

### Overview
- **Source**: HuggingFace `wikitext` (`wikitext-103-raw-v1`)
- **Size**: ~500MB, ~100M tokens
- **Format**: HuggingFace Dataset
- **Task**: Clean Wikipedia text for controlled experiments
- **Splits**: train, validation, test

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
for item in ds:
    text = item["text"]
```

## Dataset 3: Tiny Shakespeare

### Overview
- **Source**: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- **Size**: ~1.1MB, character-level
- **Format**: Plain text
- **Task**: Small naturalistic text for quick experiments and debugging
- **Already downloaded**: `datasets/tiny_shakespeare.txt`

### Loading
```python
with open("datasets/tiny_shakespeare.txt") as f:
    text = f.read()
```

## Dataset 4: Random Token Sequences (Baseline)

### Overview
- **Source**: Generated programmatically
- **Task**: Random token sequences for reproducing classic induction head detection
  (Olsson et al. 2022 methodology). Used as baseline comparison against naturalistic data.

### Generation
```python
import torch
# Generate repeated random sequences for induction head detection
seq_len = 256
vocab_size = 50257  # GPT-2 vocab
half = seq_len // 2
random_half = torch.randint(0, vocab_size, (batch_size, half))
repeated_seq = torch.cat([random_half, random_half], dim=1)
```

## Notes
- OpenWebText and WikiText are the primary datasets for naturalistic experiments
- Random token sequences serve as the baseline (replicating Olsson et al. methodology)
- Tiny Shakespeare is useful for quick iteration and debugging
- All HuggingFace datasets support streaming to avoid large downloads
