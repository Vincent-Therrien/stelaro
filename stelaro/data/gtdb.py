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
        force: If `True`, install even if the files are already installed.
    """
    stelaro_rust.install("gtdb", "trees", path, force)


def install_taxonomy(path: str, force: bool = False) -> None:
    """Download and decompress taxonomic information for bacteria and archaea.

    Args:
        path: Directory in which to install data.
        force: If `True`, install even if the files are already installed.
    """
    stelaro_rust.install("gtdb", "taxonomy", path, force)
