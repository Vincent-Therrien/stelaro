"""GTDB data analysis module.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: April 2025
    - License: MIT
"""

import os
import stelaro.stelaro as stelaro_rust


def install_trees(path: str, force: bool = False) -> None:
    """Download and decompress phylogenetic trees.

    Args:
        path: Directory in which to install data.
    """
    stelaro_rust.install("gtdb", "trees", path, force)
