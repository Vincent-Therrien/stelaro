"""Data transformation testing module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025.
    - License: MIT
"""

import numpy as np
import os
import stelaro.stelaro as stelaro_rs

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)


def count_overlapping(sequence, subsequence, k):
    return sum(
        subsequence == sequence[i:i+k] for i in range(
            len(sequence) - (k - 1)
        )
    )


def test_kmer_profiling():
    os.chdir(dname)
    with open("../data/test_sequence_A.1.fna", "r") as f:
        sequence_a = "".join(f.read().split("\n")[1:]).strip()
    with open("../data/test_sequence_B.1.fna", "r") as f:
        sequence_b = "".join(f.read().split("\n")[1:]).strip()
    for k in (3, 4, ):
        kmers = stelaro_rs.profile_kmer(
            "../data/test_index.tsv",
            "../data/",
            k
        )
        assert len(kmers) == 4**k, "Unexpected K-mer count."
        unique = set(list(kmers.keys()))
        assert len(unique) == len(kmers), "Non-unique K-mers."
        for kmer, count in kmers.items():
            expected_count = sum(
                (
                    count_overlapping(sequence_a, kmer, k),
                    count_overlapping(sequence_b, kmer, k),
                )
            )
            assert expected_count == count, (
                f"Unexpected count for K-mer {kmer}."
            )
