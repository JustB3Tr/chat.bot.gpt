import argparse
from pathlib import Path

import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from src.chatbot.config import DataConfig, FullConfig, ModelConfig, PathsConfig, TrainingConfig
from src.chatbot.data import group_texts, load_text_datasets, tokenize_dataset
from src.chatbot.modeling import build_model
from src.chatbot.tokenizer import train_tokenizer


def load_config(path: str) -> FullConfig:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = ModelConfig(**cfg["model"])
    data_cfg = DataConfig(**cfg["data"])
    training_cfg = TrainingConfig(**cfg["training"])
    paths_cfg = PathsConfig(**cfg["paths"])
    return FullConfig(model=model_cfg, data=data_cfg, training=training_cfg, paths=paths_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom decoder-only LLM from scratch.")
    parser.add_argument("--config", type=str, default="configs/small.yaml", help="Path to YAML config.")
    parser.add_argument("--train-files", nargs="+", required=True, help="Training text files.")
    parser.add_argument("--eval-files", nargs="+", help="Evaluation text files.")
    parser.add_argument("--tokenizer-dir", type=str, help="Existing tokenizer directory (tokenizer.json).")
    parser.add_argument("--output-dir", type=str, help="Override output directory for checkpoints.")
    parser.add_argument("--resume-from", type=str, help="Checkpoint path to resume from.")
    parser.add_argument("--save-samples", type=int, default=0, help="Optional: number of eval samples to save for sanity.")
    # Common overrides to tweak without editing YAML
    parser.add_argument("--block-size", type=int, help="Override sequence length.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--max-steps", type=int, help="Override max training steps.")
    parser.add_argument("--warmup-steps", type=int, help="Override warmup steps.")
    parser.add_argument("--per-device-train-batch-size", type=int, help="Override per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, help="Override per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override gradient accumulation steps.")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay.")
    parser.add_argument("--save-steps", type=int, help="Override save frequency.")
    parser.add_argument("--eval-steps", type=int, help="Override eval frequency.")
    return parser.parse_args()


def prepare_tokenizer(args: argparse.Namespace, cfg: FullConfig):
    tokenizer_dir = args.tokenizer_dir or cfg.paths.tokenizer_dir
    tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        train_tokenizer(
            data_files=args.train_files + (args.eval_files or []),
            output_dir=tokenizer_dir,
            vocab_size=cfg.model.vocab_size,
        )

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
        }
    )
    return tokenizer


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.block_size:
        cfg.data.block_size = args.block_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.warmup_steps:
        cfg.training.warmup_steps = args.warmup_steps
    if args.per_device_train_batch_size:
        cfg.training.per_device_train_batch_size = args.per_device_train_batch_size
    if args.per_device_eval_batch_size:
        cfg.training.per_device_eval_batch_size = args.per_device_eval_batch_size
    if args.gradient_accumulation_steps:
        cfg.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.save_steps:
        cfg.training.save_steps = args.save_steps
    if args.eval_steps:
        cfg.training.eval_steps = args.eval_steps

    set_seed(cfg.data.seed)

    if cfg.data.block_size > cfg.model.n_positions:
        raise ValueError(f"block_size ({cfg.data.block_size}) cannot exceed n_positions ({cfg.model.n_positions})")

    tokenizer = prepare_tokenizer(args, cfg)

    train_ds, eval_ds = load_text_datasets(args.train_files, args.eval_files)
    train_ds = tokenize_dataset(train_ds, tokenizer, cfg.data.block_size, cfg.data.num_workers)
    train_ds = group_texts(train_ds, cfg.data.block_size, cfg.data.num_workers)

    if eval_ds is not None:
        eval_ds = tokenize_dataset(eval_ds, tokenizer, cfg.data.block_size, cfg.data.num_workers)
        eval_ds = group_texts(eval_ds, cfg.data.block_size, cfg.data.num_workers)
        if args.save_samples > 0:
            eval_ds = eval_ds.select(range(min(args.save_samples, len(eval_ds))))

    model = build_model(cfg.model)
    model.resize_token_embeddings(len(tokenizer))

    output_dir = args.output_dir or cfg.paths.output_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        max_steps=cfg.training.max_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_total_limit=3,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
