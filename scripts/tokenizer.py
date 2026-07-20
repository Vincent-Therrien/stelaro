import argparse
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from random import random
from typing import Iterator


def decode_trimer(encoded_sequence: list[int]) -> str:
    BITS_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    decoded_chars = []

    for val in encoded_sequence:
        if not (0 <= val <= 63):
            raise ValueError("Encoded trimer integers must be between 0 and 63.")

        # Extract the three nucleotides from right to left
        base3 = BITS_TO_BASE[val & 3]          # Get lowest 2 bits
        base2 = BITS_TO_BASE[(val >> 2) & 3]   # Shift 2 bits, get next 2 bits
        base1 = BITS_TO_BASE[(val >> 4) & 3]   # Shift 4 bits, get highest 2 bits

        decoded_chars.extend([base1, base2, base3])

    return "".join(decoded_chars)


def tokenize(filename, output, n, maximum):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    n = 0
    x = np.load(filename)
    rng = np.random.default_rng()
    random_row_indices = rng.permutation(x.shape[0])

    def line_iterator():
        nonlocal n
        for idx in random_row_indices:
            n += 1
            if n % 10_000 == 0:
                print(f"{n:_}")
            if n > maximum:
                return
            sequence = x[idx]
            yield decode_trimer(sequence)

    overlapping_generator = line_iterator()
    tokenizer.train_from_iterator(overlapping_generator, trainer)
    tokenizer.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model benchmarking."
    )
    parser.add_argument("--file", type=str, help="Path to the file containing training data (trimer sequences).")
    parser.add_argument("--output", type=str, help="Filepath of the result file.")
    parser.add_argument("--vocabulary_size", type=int, help="Size of the output vocabulary.")
    parser.add_argument("--max", type=int, help="Maximum number of sequences.")
    args = parser.parse_args()
    tokenize(args.file, args.output, args.vocabulary_size, args.max)
