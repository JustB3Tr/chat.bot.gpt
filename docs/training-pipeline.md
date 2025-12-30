# Training Pipeline (Custom-from-Scratch)

This document describes the reference training flow for building a **fully custom** decoder-only LLM from scratch (no third-party base weights), along with tokenizer training and data preparation.

## Overview
- **Tokenizer:** Train a BPE tokenizer on your curated corpus with domain-appropriate special tokens.
- **Model:** Decoder-only Transformer (GPT-style) with configurable depth/width; trained from random init.
- **Data:** Plain-text shards; split into train/validation; streaming supported.
- **Trainer:** Hugging Face `Trainer`/`Accelerate` for distributed training; supports gradient accumulation, mixed precision, and checkpointing.
- **Outputs:** Tokenizer files, model checkpoints, and logs/metrics.

## Directory Layout
- `data/` — your raw/sharded text files (not tracked in git). Use `train.txt`/`val.txt` or glob patterns.
- `artifacts/tokenizer/` — saved tokenizer JSON/vocab files.
- `artifacts/checkpoints/` — model checkpoints.

## Quickstart
1. **Install deps**
   ```bash
   pip install -r requirements.txt
   # If PyTorch wheels are blocked in your network, download the correct torch wheel
   # from https://download.pytorch.org/ and install via:
   # pip install torch-<ver>-cp<pyver>-<platform>.whl
   # If outbound access is blocked, provide the wheel in ./deps and run:
   # pip install --no-index --find-links ./deps torch-<ver>-cp<pyver>-<platform>.whl
   ```
2. **Train tokenizer**
   ```bash
   python -m src.chatbot.tokenizer \\
     --data-files \"data/train.txt\" \"data/val.txt\" \\
     --vocab-size 32000 \\
     --output-dir artifacts/tokenizer
   ```
3. **Train model**
   ```bash
   python scripts/train.py \\
     --train-files \"data/train.txt\" \\
     --eval-files \"data/val.txt\" \\
     --tokenizer-dir artifacts/tokenizer \\
     --output-dir artifacts/checkpoints \\
     --config configs/small.yaml
   ```
   Common overrides (no YAML edit needed):
   ```bash
   python scripts/train.py \\
     --train-files \"data/train.txt\" \\
     --eval-files \"data/val.txt\" \\
     --tokenizer-dir artifacts/tokenizer \\
     --output-dir artifacts/checkpoints \\
     --config configs/small.yaml \\
     --block-size 1024 \\
     --learning-rate 2e-4 \\
     --per-device-train-batch-size 4 \\
     --gradient-accumulation-steps 16 \\
     --max-steps 20000 \\
     --warmup-steps 1000
   ```

## Configs
See `configs/small.yaml` for a starting point. Key sections:
- `model`: layers, heads, hidden size, context length, vocab size.
- `training`: batch sizes, learning rate schedule, warmup, max steps/epochs.
- `data`: sequence length, num workers, shuffle buffer.

## Personalization Hooks
- Add **user/persona embeddings** downstream by extending the model input (e.g., prefix tokens or adapter inputs). The current scaffold focuses on base pretraining; finetuning can condition on persona vectors stored in memory/checkpoints.

## Notes
- This is a reference scaffold; wire it into your orchestrator/memory stack as you proceed.
- Use mixed precision (`bf16` if hardware permits) and gradient checkpointing for larger configs.
- For larger runs, prefer `accelerate launch` to distribute across nodes/GPUs.
- Ensure `block_size <= n_positions` in your config/overrides; the script enforces this.
