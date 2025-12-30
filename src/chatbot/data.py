from typing import Callable, Iterable, Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def load_text_datasets(
    train_files: list[str],
    eval_files: Optional[list[str]],
) -> tuple[Dataset, Optional[Dataset]]:
    train_ds = load_dataset("text", data_files=train_files, split="train")
    eval_ds = None
    if eval_files:
        eval_ds = load_dataset("text", data_files=eval_files, split="train")
    return train_ds, eval_ds


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    num_workers: int = 4,
) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def group_texts(
    dataset: Dataset,
    block_size: int,
    num_workers: int = 4,
) -> Dataset:
    def chunk_examples(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = sum(batch["input_ids"], [])
        total_length = (len(concatenated) // block_size) * block_size
        input_ids = [
            concatenated[i : i + block_size] for i in range(0, total_length, block_size)
        ]
        attention_mask = [[1] * block_size for _ in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return dataset.map(
        chunk_examples,
        batched=True,
        num_proc=num_workers,
    )
