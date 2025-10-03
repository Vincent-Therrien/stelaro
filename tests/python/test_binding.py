"""Rust/Python binding test module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import numpy as np
import os
import stelaro.stelaro as stelaro_rs

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)


def test_encoding():
    os.chdir(dname)
    encodings, identifiers = stelaro_rs.synthetic_sample(
        "../data/test_index.tsv",
        "../data/",
        reads=50,
        length=25,
        length_deviation=0,
        indels=0,
        indels_deviation=0,
        encoding="onehot"
    )
    assert set(identifiers) == set(
        (
            "test_sequence_A.1.fna",
            "test_sequence_B.1.fna"
        )
    ), "Unexpected genome identifiers."
    with open("../data/test_sequence_A.1.fna", "r") as f:
        sequence_a = "".join(f.read().split("\n")[1:]).strip()
    with open("../data/test_sequence_B.1.fna", "r") as f:
        sequence_b = "".join(f.read().split("\n")[1:]).strip()
    sequences = stelaro_rs.decode(encodings, "onehot")
    # Validate presence in the right genomes.
    for sequence, identifier in zip(sequences, identifiers):
        sequence = sequence[:-sequence.count("N")]  # Strip the padding.
        if identifier == "test_sequence_A.1.fna":
            assert sequence in sequence_a
        else:
            assert sequence in sequence_b
    # Validate the encoding.
    for sequence, encoding in zip(sequences, encodings):
        python_decoding = ""
        for element in encoding:
            if element[0]:
                python_decoding += "A"
            elif element[1]:
                python_decoding += "C"
            elif element[2]:
                python_decoding += "G"
            elif element[3]:
                python_decoding += "T"
            else:
                python_decoding += "N"
        assert python_decoding == sequence
