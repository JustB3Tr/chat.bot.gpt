from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_positions: int
    rotary_pct: float = 1.0
    dropout: float = 0.0


@dataclass
class DataConfig:
    block_size: int
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    lr_scheduler_type: str
    max_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


@dataclass
class PathsConfig:
    tokenizer_dir: str
    output_dir: str
    train_files: Optional[list[str]] = None
    eval_files: Optional[list[str]] = None


@dataclass
class FullConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    paths: PathsConfig
