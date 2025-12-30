import argparse
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def iter_files(paths: list[str]) -> Iterable[str]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                yield line.strip()


def train_tokenizer(data_files: list[str], output_dir: str, vocab_size: int = 32000) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train_from_iterator(iter_files(data_files), trainer=trainer)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path / "tokenizer.json"))
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from scratch.")
    parser.add_argument(
        "--data-files",
        nargs="+",
        required=True,
        help="Text files to use for tokenizer training.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size for tokenizer.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the trained tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tokenizer(
        data_files=args.data_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
