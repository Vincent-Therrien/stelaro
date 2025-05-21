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

__all__ = ["ncbi", ]


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
