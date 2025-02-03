"""Rust/Python input / output test module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: October 2024
    - License: MIT
"""

import os
import stelaro.stelaro as stelaro_rs

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

FASTA = "../data/nucleotide_sequence.fasta"
FASTQ = "../data/nucleotide_sequence.fastq"


def test_read_fasta():
    os.chdir(dname)
    result = stelaro_rs.read_fasta(FASTA)["obj"]
    assert len(result) == 2, "unexpected count."
    assert result[0][1] == "AACCGGTTAACCGGTT", "Unexpected sequence"
    assert result[1][1] == "AACCGGTT", "Unexpected sequence"


def test_read_fastq():
    os.chdir(dname)
    result = stelaro_rs.read_fastq(FASTQ)["obj"]
    print(result)
    assert len(result) == 3
    assert result[0][1] == "AACCGGTTAACCGGTT", "Unexpected sequence"
    quality = [int(b) for b in result[0][2]]
    assert quality == [
        0, 0, 0, 93, 93, 93, 93, 93, 0, 0, 0, 93, 93, 93, 93, 93
    ]
