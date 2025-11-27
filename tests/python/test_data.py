"""Data testing module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: August 2025.
    - License: MIT
"""

from stelaro.data import format
from torch import tensor


def test_data_format():
    sequence = "ACGTAGGTATCGATGCTAGTCGATGATCTTATGAAG"
    # Tetramer.
    tetramer = format.encode_tetramer(sequence)
    assert len(tetramer) == len(sequence) / 4
    batch = tensor(tetramer).unsqueeze(dim=0)
    # Digit encoding.
    digits = format.tetramer_batch_to_digits(batch)[0]
    assert len(digits) == len(sequence)
    reconstructed_sequence = format.decode_digits(digits)
    assert reconstructed_sequence == sequence
    # Onehot encoding.
    onehot = format.tetramer_batch_to_onehot(batch)[0]
    assert len(onehot) == len(sequence)
    reconstructed_sequence = format.decode_onehot(onehot)
    assert reconstructed_sequence == sequence
    # Codon encoding.
    codons = format.to_codons(batch)[0]
    assert len(codons) == len(sequence) / 3
    reconstructed_sequence = format.decode_codons(codons)
    assert reconstructed_sequence == sequence

