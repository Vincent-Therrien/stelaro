"""Data transformation testing module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: June 2025.
    - License: MIT
"""

import numpy as np
import os
import stelaro.stelaro as stelaro_rs
from stelaro.transform import kmer

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


def test_extract_n_non_overlapping():
    profile = {
        "AAAA": 10,
        "CAAA": 9,
        "CCAA": 8,
        "CCCA": 7,
        "GGGG": 6,
        "TTTT": 5
    }
    assert kmer.extract(profile, 6) == list(profile.keys())
    assert kmer.extract(profile, 6, 1) == ["AAAA", "CCAA", "GGGG", "TTTT"]
    assert kmer.extract(profile, 3, 1) == ["AAAA", "CCAA", "GGGG"]
    assert kmer.extract(profile, 6, 2) == ["AAAA", "CCCA", "GGGG", "TTTT"]
    assert kmer.extract(profile, 3, 2) == ["AAAA", "CCCA", "GGGG"]


def test_compress():
    sequence = "AAAAGGGGAAAA"
    scheme = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "AAAA": 4,
        "AC": 5,
        "CG": 6
    }
    compression = kmer.compress(sequence, scheme)
    assert compression == [4, 2, 2, 2, 2, 4]


# scheme = {"N": 0, "NNNNNNNN": 1, "A": 2, "C": 3, "G": 4, "T": 5}
# sizes = {4: 250}
# genome_size = data.get_index_size(INDEX)
# print(f"Genome database length: {genome_size}")
# for K in sizes:
#     profile = kmer.profile(INDEX, DIRECTORY, K)
#     for i in (1, 8, 16, 32):
#         keys = kmer.extract(profile, i, K // 2)
#         n = 0
#         elements = []
#         for key in keys:
#             n += profile[key]
#             elements.append((key, profile[key]))
#         print(f"K = {K}, {i} highest: {(100.0 * n * K / genome_size):.2f} %")
#     if K in sizes:
#         overlap = K // 2 if K > 4 else None
#         keys = kmer.extract(profile, sizes[K] * 4, overlap)
#         for i in range(sizes[K]):
#             scheme[keys[i]] = len(scheme)
# print(scheme)
