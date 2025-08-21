"""
    Format DNA sequences into useful formats.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: August 2025
    - License: MIT
"""

from torch import Tensor, tensor, stack, zeros_like
from torch.nn.functional import one_hot


BASE_TO_BITS = {
    'A': 0b00,
    'C': 0b01,
    'G': 0b10,
    'T': 0b11,
}


def encode_tetramer(sequence: str) -> list[int]:
    """Convert a nucleotide sequence into tetramer encoding.

    Args:
        sequence: Nucleotide sequence.

    Returns: 4-mer encoded sequence.
    """
    if any(base not in 'ACGT' for base in sequence):
        raise ValueError("Input must be a 4-character string containing only A, C, G, T.")

    def encode_four_nucleotides(tetramer):
        result = 0
        for base in tetramer:
            result = (result << 2) | BASE_TO_BITS[base]
        return result

    tetramers = [sequence[i:i+4] for i in range(0, len(sequence), 4)]
    return [encode_four_nucleotides(t) for t in tetramers]


def decode_tetramer(sequence: list[int]) -> str:
    """Convert a tetramer encoding into a nucleotide sequence.

    Args:
        sequence: 4-mer encoded sequence.

    Returns: Nucleotide sequence.
    """
    bits_to_base = {
        0b00: 'A',
        0b01: 'C',
        0b10: 'G',
        0b11: 'T'
    }

    def decode_integer(integer):
        tetramer = ''
        for shift in (6, 4, 2, 0):
            two_bits = (integer >> shift) & 0b11
            tetramer += bits_to_base[two_bits]
        return tetramer

    return "".join([decode_integer(i) for i in sequence])


def tetramer_batch_to_digits(batch: Tensor) -> list[list[int]]:
    """Convert a batch of tetramer-encoded reads into a digit batch.

    Args:
        batch: Tensor of shape (batch_size, sequence_length)

    Returns: Digit-encoded batch.
    """
    B, N = batch.shape
    shifts = tensor([6, 4, 2, 0], device=batch.device).view(1, 1, 4)
    x_expanded = batch.unsqueeze(-1)
    tokens = (x_expanded >> shifts) & 0b11
    digits = tokens.view(B, N * 4)
    return digits


def decode_digits(digits: list[int]) -> str:
    """Convert a digits encoding into a nucleotide sequence.

    Args:
        digits: digits encoded sequence.

    Returns: Nucleotide sequence.
    """
    return "".join(["ACGT"[digit] for digit in digits])


def tetramer_batch_to_onehot(batch: Tensor) -> list[list[int]]:
    """Convert a batch of tetramer-encoded reads into a onehot batch.

    Args:
        batch: Tensor of shape (batch_size, sequence_length)

    Returns: Onehot-encoded batch.
    """
    digits = tetramer_batch_to_digits(batch)
    return one_hot(digits)


def decode_onehot(onehot: list[int]) -> str:
    """Convert a digits encoding into a nucleotide sequence.

    Args:
        digits: digits encoded sequence.

    Returns: Nucleotide sequence.
    """
    digits = onehot.argmax(dim=1)
    return "".join(["ACGT"[digit] for digit in digits])


def tetramer_batch_to_codons(batch: Tensor) -> list[list[int]]:
    """Convert a batch of tetramer-encoded reads into a 3-mer batch.

    Args:
        batch: Tensor of shape (batch_size, sequence_length)

    Returns: 3-mer-encoded batch.
    """
    digits = tetramer_batch_to_digits(batch)
    B, N = digits.shape
    codons = digits.view(B, N // 3, 3)
    powers = tensor([16, 4, 1], device=batch.device, dtype=batch.dtype)
    codons_encoded = (codons * powers).sum(dim=-1)
    return codons_encoded


def decode_codons(codons: list[int]) -> str:
    """Convert a codon encoding into a nucleotide sequence.

    Args:
        codon: codon encoded sequence.

    Returns: Nucleotide sequence.
    """
    batch = codons.unsqueeze(dim=0)
    B, N = batch.shape
    shifts = tensor([4, 2, 0], device=batch.device).view(1, 1, 3)
    x_expanded = batch.unsqueeze(-1)
    tokens = (x_expanded >> shifts) & 0b11
    onehot = tokens.view(B, N * 3)[0]
    print(onehot)
    return "".join (["ACGT"[digit] for digit in onehot])
