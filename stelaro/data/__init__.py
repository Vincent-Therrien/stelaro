"""
    The modules in this directory serve as (1) bindings for the `data` Rust
    module that installs and processes data and (2) a Python module for data
    visualization.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: November 2024
    - License: MIT
"""

import stelaro.stelaro as stelaro_rust

__all__ = ["ncbi", "gtdb", ]


def synthetic_metagenome(
        src: str,
        genomes: str,
        dst: str,
        reads: int,
        length: int,
        length_deviation: int = 0,
        indels: int = 0,
        indels_deviation: int = 0
        ) -> None:
    stelaro_rust.synthetic_metagenome(
        src,
        genomes,
        dst,
        reads,
        length,
        length_deviation,
        indels,
        indels_deviation
    )


def get_index_size(index: str) -> int:
    """Get the size of the genomes stored at an index."""
    total = 0
    with open(index, "r") as f:
        next(f)  # Skip the header.
        for line in f:
            fields = line.strip().split("\t")
            size = int(fields[-1])
            total += size
    return total


def read_genome(file: str) -> list[str]:
    """Read a reference genome."""
    return stelaro_rust.read_fasta(file)["obj"]
